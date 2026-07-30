#!/usr/bin/env python3
"""Run the trained DINOv2 road-shoulder model on Autoware's live CARLA camera feed.

Subscribes to a camera topic published by ``autoware_carla_interface`` (default:
``/sensing/camera/CAM_FRONT/image_raw``), runs the same 3-class
(travel_lane / shoulder / non_driveable) segmentation model used in
``shoulder_detection_carla.py``, and publishes:

  - ``~/overlay_image``   (sensor_msgs/Image, bgr8)   colourised overlay, for RViz
  - ``~/class_mask``      (sensor_msgs/Image, mono8)  raw class-index mask (0/1/2),
                          consumed by carla_ground_projection_node -- no second
                          inference pass needed there
  - ``~/class_fractions`` (std_msgs/Float32MultiArray) [travel_lane, shoulder,
                          non_driveable] fractions of the frame, in that fixed order

This node needs the project's dedicated venv (torch/transformers/albumentations) --
see the workspace README. It does NOT talk to CARLA directly; it only touches ROS
topics, so it works the same whether the ego vehicle came from autoware_carla_interface
or any other source that publishes a compatible image topic.
"""
import os
import sys

# The DINOv2 backbone is loaded via `transformers`, which by default checks
# huggingface.co for updates on every from_pretrained() call before falling
# back to the local cache. This machine doesn't have reliable internet access,
# so that check just burns the retry/backoff delay (up to ~30s) every startup
# before falling back anyway. The weights are already fully cached locally
# (~/.cache/huggingface/hub/models--facebook--dinov2-base), so go straight to
# offline/cache-only lookup instead. Must be set before `transformers` is
# imported (below, transitively via shoulder_detection_project).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# This node needs torch/transformers/albumentations, which only exist in this
# workspace's dedicated venv (see ../../README.md) -- but `ros2 run`/
# `ros2 launch` always execute the installed script via the shebang colcon
# baked in at build time, which is plain system python3 (colcon itself is a
# system-installed tool, so it builds with its own interpreter regardless of
# which venv was active in your shell when you ran `colcon build`).
#
# Detect this by trying the actual import, not by comparing sys.executable:
# a venv's bin/python3 is normally just a symlink to the same system
# interpreter binary, so realpath(sys.executable) is identical inside and
# outside the venv -- what differs is sys.path, so the import itself is the
# only reliable test. Probing for `albumentations` rather than `torch`: this
# machine's system python3 has a stray, unrelated, ancient system-wide torch
# (1.8.0a0 from /usr/lib/python3/dist-packages -- some apt package's
# dependency), so a plain `import torch` succeeds even outside the venv and
# would never trigger the re-exec; albumentations is confirmed venv-only.
try:
    import albumentations  # noqa: F401
except ImportError:
    _VENV_PYTHON = os.environ.get(
        "SHOULDER_DETECTION_VENV_PYTHON",
        os.path.expanduser("~/shoulder_detection_ws/.venv/bin/python3"),
    )
    if not os.path.exists(_VENV_PYTHON):
        raise RuntimeError(
            f"torch is not importable under {sys.executable}, and no venv was found at "
            f"{_VENV_PYTHON}. Run ~/shoulder_detection_ws/setup_venv.sh, or set the "
            f"SHOULDER_DETECTION_VENV_PYTHON environment variable to the right interpreter."
        )
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import Float32MultiArray  # noqa: E402

# --- make the shoulder_detection_project package importable (same convention
# as shoulder_detection_carla.py) ---
SHOULDER_PROJECT_ROOT = os.environ.get(
    "SHOULDER_PROJECT_ROOT", os.path.expanduser("~/shoulder_detection_project")
)
if not os.path.isdir(SHOULDER_PROJECT_ROOT):
    raise RuntimeError(
        f"shoulder_detection_project not found at {SHOULDER_PROJECT_ROOT}. "
        f"Set the SHOULDER_PROJECT_ROOT environment variable to its path."
    )
sys.path.insert(0, SHOULDER_PROJECT_ROOT)

from src.infer import CLASS_NAMES, ShoulderInference  # noqa: E402

# CLASS_NAMES = {0: "travel_lane", 1: "shoulder", 2: "non_driveable"} -- fixed
# publish order for the class_fractions message, index-matched to this dict.
FRACTION_ORDER = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

DEFAULT_CHECKPOINT = os.path.join(
    SHOULDER_PROJECT_ROOT, "models", "checkpoints", "sne_shoulder", "best_shoulder_iou.pth"
)


class ShoulderDetectionNode(Node):
    def __init__(self):
        super().__init__("shoulder_detection_node")

        self.declare_parameter("image_topic", "/sensing/camera/CAM_FRONT/image_raw")
        self.declare_parameter("overlay_topic", "~/overlay_image")
        self.declare_parameter("mask_topic", "~/class_mask")
        self.declare_parameter("fractions_topic", "~/class_fractions")
        self.declare_parameter("arch", "dinov2")
        self.declare_parameter("checkpoint", "")  # empty string = use DEFAULT_CHECKPOINT below
        self.declare_parameter("half", False)
        self.declare_parameter("alpha", 0.45)
        self.declare_parameter("log_every_n", 60)

        # Shoulder-only overlay: same cv2.addWeighted blend as the full
        # overlay above, but applied only where the mask is `shoulder_class`
        # -- travel_lane/non_driveable pixels are left as the original camera
        # image, untouched. Viewed in RViz (Image display) and in the live
        # shoulder_overlay_viewer_node window.
        self.declare_parameter("shoulder_overlay_topic", "~/shoulder_only_overlay_image")
        self.declare_parameter("shoulder_overlay_class", 1)
        # Same BGR value as COLORS_BGR[1] in shoulder_detection_project/src/infer.py
        # and VIZ_BGR[1] in scripts/cvat_seg_to_masks.py -- kept consistent with
        # the CVAT annotation color across the whole project, not an independent
        # visualization choice.
        self.declare_parameter("shoulder_overlay_color_bgr", [245, 61, 61])
        self.declare_parameter("shoulder_overlay_alpha", 0.55)

        arch = self.get_parameter("arch").value
        checkpoint = self.get_parameter("checkpoint").value or DEFAULT_CHECKPOINT
        half = self.get_parameter("half").value
        self.alpha = float(self.get_parameter("alpha").value)
        self.log_every_n = int(self.get_parameter("log_every_n").value)
        self.shoulder_overlay_class = int(self.get_parameter("shoulder_overlay_class").value)
        self.shoulder_overlay_color_bgr = tuple(
            int(c) for c in self.get_parameter("shoulder_overlay_color_bgr").value
        )
        self.shoulder_overlay_alpha = float(self.get_parameter("shoulder_overlay_alpha").value)

        self.get_logger().info(f"loading {arch} model from {checkpoint} ...")
        self.engine = ShoulderInference(arch, checkpoint, half=half)
        self.get_logger().info(f"model ready on {self.engine.device}")

        self.bridge = CvBridge()
        self.frame_idx = 0

        image_topic = self.get_parameter("image_topic").value
        self.overlay_pub = self.create_publisher(Image, self.get_parameter("overlay_topic").value, 1)
        self.shoulder_overlay_pub = self.create_publisher(
            Image, self.get_parameter("shoulder_overlay_topic").value, 1
        )
        self.mask_pub = self.create_publisher(Image, self.get_parameter("mask_topic").value, 1)
        self.fractions_pub = self.create_publisher(
            Float32MultiArray, self.get_parameter("fractions_topic").value, 1
        )
        self.sub = self.create_subscription(Image, image_topic, self._on_image, 1)
        self.get_logger().info(f"subscribed to {image_topic}")

    def _on_image(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        t0 = time.time()
        mask = self.engine.predict(frame_bgr)
        dt_ms = (time.time() - t0) * 1000.0

        overlay_bgr = self.engine.overlay(frame_bgr, mask, alpha=self.alpha)
        fractions = self.engine.class_fractions(mask)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        overlay_msg.header = msg.header
        self.overlay_pub.publish(overlay_msg)

        shoulder_mask = mask == self.shoulder_overlay_class
        colour = np.empty_like(frame_bgr)
        colour[:] = self.shoulder_overlay_color_bgr
        tinted = cv2.addWeighted(frame_bgr, 1 - self.shoulder_overlay_alpha, colour, self.shoulder_overlay_alpha, 0)
        shoulder_overlay_bgr = frame_bgr.copy()
        shoulder_overlay_bgr[shoulder_mask] = tinted[shoulder_mask]
        shoulder_overlay_msg = self.bridge.cv2_to_imgmsg(shoulder_overlay_bgr, encoding="bgr8")
        shoulder_overlay_msg.header = msg.header
        self.shoulder_overlay_pub.publish(shoulder_overlay_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(mask.astype(np.uint8), encoding="mono8")
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)

        fractions_msg = Float32MultiArray()
        fractions_msg.data = [float(fractions[name]) for name in FRACTION_ORDER]
        self.fractions_pub.publish(fractions_msg)

        if self.frame_idx % self.log_every_n == 0:
            fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0
            self.get_logger().info(
                f"frame {self.frame_idx:6d} | {dt_ms:5.1f} ms ({fps:4.1f} FPS) | "
                f"shoulder {fractions['shoulder'] * 100:5.1f}%"
            )
        self.frame_idx += 1


def main():
    rclpy.init()
    node = ShoulderDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
