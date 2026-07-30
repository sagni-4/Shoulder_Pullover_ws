#!/usr/bin/env python3
"""Live on-screen window(s) showing a shoulder_detection_node overlay image
next to the CARLA game window, since CARLA's own 3D viewport can't be
post-processed with a pixel overlay from outside (its Python API only
exposes unlit debug draw primitives -- lines/points/boxes -- with no
filled-polygon or flat image-blend capability at all).

This is deliberately the simplest possible node: it does not run the model,
does not touch the CARLA API, and does not do any ray-casting/ground
projection. It just subscribes to an already-computed overlay image topic
and displays it with cv2.imshow, optionally with a small HUD text bar (FPS +
class fractions) drawn on top of the *displayed* frame only (never written
back to any published topic).

Two independent instances of this node are launched by
shoulder_detection.launch.xml, each independently toggleable
(`enable_shoulder_only_viewer_window` / `enable_full_class_viewer_window`),
so either or both windows can be shown:
  - "shoulder only": subscribes to shoulder_detection_node's
    `~/shoulder_only_overlay_image` -- flat light-highlight tint on just the
    shoulder pixels, everything else left as the original camera image.
  - "full classes": subscribes to `~/overlay_image` -- all 3 classes tinted
    (travel_lane/shoulder/non_driveable), matching the reference video's
    look (shoulder_detection_carla.py's own pygame HUD demo).

On FPS: the display rate here is capped by shoulder_detection_node's model
inference rate feeding the subscribed topic (this node does no inference of
its own and adds negligible per-frame overhead -- an image decode + imshow +
waitKey(1), all sub-millisecond for a single frame). If the window feels
slow, check `nvidia-smi` before assuming a bug here: on this machine the GPU
is shared by CARLA's own rendering (especially once run without
-RenderOffScreen), Autoware's perception/planning component containers, and
this model, all on one card -- 95% GPU utilization with all of those
concurrent was directly observed to correlate with a roughly 2x slower
inference rate than shoulder_detection_carla.py's standalone reference
script achieves running alongside just CARLA alone. The in-window FPS
counter (`show_hud`) reports the true end-to-end rate so this is visible
directly rather than guessed at.

History: earlier versions of this package (carla_ground_projection_node, now
removed) tried to draw a 3D-projected approximation of the shoulder region
directly into CARLA's world (`world.debug.draw_line`) and as an RViz
`Marker`. Both looked poor -- coarse, blocky, and CARLA's version was prone
to a "blown-out white blob" bug from unstable near-horizon ray/ground-plane
math -- and neither could match the flat, pixel-accurate look of a genuine
2D image overlay. Replaced with this much simpler node instead of continuing
to chase 3D-projection fidelity.
"""
from collections import deque

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# Matches shoulder_detection_node's FRACTION_ORDER (sorted CLASS_NAMES keys):
# [travel_lane, shoulder, non_driveable]. Fixed by that node's publish
# contract, not re-derived here to avoid an import into the venv-only model code.
FRACTION_LABELS = ["travel_lane", "shoulder", "non_driveable"]


class ShoulderOverlayViewerNode(Node):
    def __init__(self):
        super().__init__("shoulder_overlay_viewer_node")

        self.declare_parameter("image_topic", "/shoulder_detection_node/shoulder_only_overlay_image")
        self.declare_parameter("window_name", "Shoulder detection (CARLA feed)")
        self.declare_parameter("show_hud", True)
        self.declare_parameter("hud_label", "")
        self.declare_parameter("fractions_topic", "/shoulder_detection_node/class_fractions")
        self.declare_parameter("fps_window", 20)  # rolling average window, in frames

        self.window_name = self.get_parameter("window_name").value
        self.show_hud = bool(self.get_parameter("show_hud").value)
        self.hud_label = self.get_parameter("hud_label").value
        self.fps_window = max(2, int(self.get_parameter("fps_window").value))

        self.bridge = CvBridge()
        self._recv_times = deque(maxlen=self.fps_window)
        self._latest_fractions = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        image_topic = self.get_parameter("image_topic").value
        self.sub = self.create_subscription(Image, image_topic, self._on_image, 1)

        if self.show_hud:
            fractions_topic = self.get_parameter("fractions_topic").value
            self.fractions_sub = self.create_subscription(
                Float32MultiArray, fractions_topic, self._on_fractions, 1
            )

        self.get_logger().info(f"subscribed to {image_topic}, showing window '{self.window_name}'")

    def _on_fractions(self, msg: Float32MultiArray):
        self._latest_fractions = msg.data

    def _current_fps(self):
        now = self.get_clock().now()
        self._recv_times.append(now)
        if len(self._recv_times) < 2:
            return 0.0
        span_sec = (self._recv_times[-1] - self._recv_times[0]).nanoseconds / 1e9
        if span_sec <= 0.0:
            return 0.0
        return (len(self._recv_times) - 1) / span_sec

    def _draw_hud(self, frame_bgr):
        fps = self._current_fps()
        parts = [self.hud_label] if self.hud_label else []
        parts.append(f"{fps:.1f} FPS")
        if self._latest_fractions is not None:
            parts.append(
                " ".join(
                    f"{label} {frac * 100:4.1f}%"
                    for label, frac in zip(FRACTION_LABELS, self._latest_fractions)
                )
            )
        text = " | ".join(parts)

        h, w = frame_bgr.shape[:2]
        bar_h = 28
        cv2.rectangle(frame_bgr, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.putText(frame_bgr, text, (6, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    def _on_image(self, msg: Image):
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.show_hud:
            frame_bgr = self._draw_hud(frame_bgr)
        cv2.imshow(self.window_name, frame_bgr)
        # Required for OpenCV's HighGUI to actually pump window/X11 events and
        # repaint -- without a periodic waitKey() call the window never
        # updates, even though imshow() itself was called every frame.
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or 'q' closes the window and exits
            self.get_logger().info("quit key pressed, shutting down")
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ShoulderOverlayViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
