# shoulder_detection_ws

Runs the trained DINOv2 road-shoulder segmentation model on Autoware's live CARLA
camera feed (instead of the standalone `shoulder_detection_carla.py` script's
direct CARLA sensor), and visualizes the result two ways:

- **RViz** -- the colourised overlay and raw class mask, published as normal
  ROS image topics.
- **CARLA** -- the detected regions projected onto the road surface and drawn
  directly in the CARLA 3D world (not a separate window), so you see it right
  on the road in the simulator itself.

This is a separate colcon workspace from `~/autoware`, kept independent because
it needs a dedicated Python virtualenv for the model's ML dependencies (torch,
transformers, ...), which don't fit into Autoware's own workspace/rosdep setup.

## Architecture

```
autoware_carla_interface (already running, part of ~/autoware)
        |
        | /sensing/camera/CAM_FRONT/image_raw  (sensor_msgs/Image)
        v
+--------------------------+
| shoulder_detection_node  |  <- needs the venv (torch, transformers, ...)
|  runs the DINOv2 model   |
+--------------------------+
        |                     \
        | ~/overlay_image       \ ~/class_mask
        | (for RViz)              (mono8 class-index image)
        v                          v
      RViz                +-------------------------------+
                           | carla_ground_projection_node |  <- lightweight,
                           |  projects mask onto the road  |     no torch needed
                           |  in the live CARLA world      |
                           +-------------------------------+
                                       |
                                       v
                              CARLA server (world.debug.draw_point)
```

Two nodes, one package, split by weight:
- `shoulder_detection_node` does the actual GPU inference -- it only touches
  ROS topics, never CARLA directly, so it's identical whether the image came
  from `autoware_carla_interface` or anything else.
- `carla_ground_projection_node` never runs the model -- it just consumes the
  already-computed mask and draws it in CARLA. It only needs `carla` +
  `cv_bridge` + `rclpy` (confirmed importable outside the venv too on this
  machine), but is built alongside the other node in the same venv for
  simplicity (one workspace, one environment, one build command).

## One-time setup

```bash
cd ~/shoulder_detection_ws
./setup_venv.sh                       # creates .venv with pinned torch/transformers/carla
source .venv/bin/activate
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

**Build with the venv active.** `colcon build` bakes the currently-active
`python3` into each installed console_scripts' shebang line. Building inside
the venv means both nodes' installed scripts point straight at the venv's
Python (which has torch, transformers, carla, *and* can still see rclpy/
cv_bridge via the inherited ROS `PYTHONPATH`) -- so `ros2 launch`/`ros2 run`
work correctly afterwards even from a plain terminal where the venv isn't
active, because the shebang does the work.

This assumes `~/shoulder_detection_project` already exists with the trained
checkpoint at `models/checkpoints/sne_shoulder/best_shoulder_iou.pth` (same
layout `shoulder_detection_carla.py` uses). Override with the
`SHOULDER_PROJECT_ROOT` environment variable if it lives elsewhere.

## Running

1. Start CARLA and the Autoware e2e_simulator with the CARLA bridge, as usual
   (see `~/carla_autoware_launch_commands.txt` / the Autoware-CARLA integration
   PDF). Confirm the ego vehicle is spawned and driving before starting this.

2. In a new terminal:
   ```bash
   source /opt/ros/humble/setup.bash
   source ~/autoware/install/setup.bash
   source ~/shoulder_detection_ws/install/setup.bash

   ros2 launch shoulder_detection_ros shoulder_detection.launch.xml
   ```

3. In RViz (the one already open from the Autoware launch, or add
   `rviz:=true` to launch a standalone one preconfigured with this package's
   displays): add/enable the `Shoulder overlay` Image display on
   `/shoulder_detection_node/overlay_image` if it isn't already there.

4. Look at the CARLA window (or the `spectator_follow`-tracked view): the
   detected travel-lane / shoulder / non-driveable regions appear as coloured
   points drawn directly on the road ahead of the vehicle, refreshed every
   frame.

### Useful launch arguments

| Argument | Default | Purpose |
|---|---|---|
| `image_topic` | `/sensing/camera/CAM_FRONT/image_raw` | Source camera topic |
| `arch` | `dinov2` | Model architecture |
| `checkpoint` | *(node default)* | Override checkpoint path |
| `half` | `false` | FP16 inference |
| `enable_carla_projection` | `true` | Set `false` to skip the in-world CARLA drawing (RViz still works) |
| `role_name` | `ego_vehicle` | Must match `ego_vehicle_role_name` used by `autoware_carla_interface` |
| `fov_deg` | `70.0` | Must match CAM_FRONT's `fov` in `sensor_mapping.yaml` -- **set to `120.0`** if you launched Autoware with `use_light_weight_sensor_mapping:=True` |
| `grid_stride_px` | `40` | Pixel spacing of the sampled projection grid (lower = denser, slower) |
| `point_life_time` | `0.3` | Seconds each drawn CARLA point persists |

Full parameter list is documented in each node's module docstring.

## Topics published (by `shoulder_detection_node`)

- `/shoulder_detection_node/overlay_image` (`sensor_msgs/Image`, bgr8) -- colourised overlay for RViz
- `/shoulder_detection_node/class_mask` (`sensor_msgs/Image`, mono8) -- raw class index per pixel (0=travel_lane, 1=shoulder, 2=non_driveable); this is what `carla_ground_projection_node` consumes
- `/shoulder_detection_node/class_fractions` (`std_msgs/Float32MultiArray`) -- `[travel_lane, shoulder, non_driveable]` fractions of the current frame, in that fixed order

## How the CARLA in-world projection works (and its limits)

`carla_ground_projection_node` treats each sampled mask pixel as a ray from
the camera (whose pose comes from the ego actor's live CARLA transform plus
its fixed mount offset, taken directly from
`carla_sensor_kit_description/config/sensor_kit_calibration.yaml`) and
intersects it with a horizontal plane at the vehicle's current ground height.
This assumes the road is **locally flat** around the vehicle -- on slopes,
hills, or banked shoulders, points farther from the camera will drift off the
true road surface. It's a visualization aid for a flat highway/test town
(matches `Town04`'s highway loop well), not a metrology tool.

## Troubleshooting

- **`ModuleNotFoundError: torch`** -- you built or ran outside the venv;
  `source ~/shoulder_detection_ws/.venv/bin/activate` before `colcon build`,
  or just re-open a terminal (the installed scripts' shebangs should handle
  it at run time regardless, once built correctly once).
- **`carla_ground_projection_node` warns "ego actor ... not found"** -- the
  Autoware/CARLA launch hasn't spawned the vehicle yet, or `role_name` doesn't
  match; it keeps retrying, no restart needed once the ego appears.
- **Points appear offset from the road** -- check `fov_deg` matches the
  active `sensor_mapping.yaml` (70.0 normal / 120.0 light-weight); a mismatch
  here directly skews the projection geometry.
