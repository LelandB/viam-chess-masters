# Viam labeled-object pick and place

Demo sentence: **When one selected, center-graspable object is visible in the
pickup zone, robot12 identifies it by shape/color label, moves it to the fixed
place pose, and returns to the observation pose.**

The application is intentionally an external Python SDK app. It does not add a
custom service to `viam-server`.

## Live resource path

```text
camera-1 -> shape-detector -> vision-segment -> Python SDK
         -> world-frame target -> builtin Motion -> arm-1 + gripper-1

camera-1 -> yellow-cylinder-detector -> SDK circle/top-depth locator -> Python SDK
```

Required resources:

- `arm-1`
- `camera-1` with point-cloud support
- `gripper-1`
- `home-pose`
- `table` with its world-frame collision geometry
- `shape-detector`
- `vision-segment`
- `yellow-cylinder-detector` and `yellow-cylinder-segment` for the upright
  yellow cylinder
- the built-in Motion service, named `builtin`

`module-reference.py` is retained only as workshop history. It is not part of
the demo runtime.

## Verified machine configuration (July 24, 2026)

The live `robot12` configuration was reduced toward the pipeline above. The
failed Hugging Face service, the black/white detector, obsolete ML model
services, unused YOLO and TensorFlow Lite modules, and leftover chess model
packages were removed. Viam configuration history remains the rollback surface.

The Viam test panel and SDK return labeled 2D detections and 3D boxes in the
`camera-1` frame. The saved configuration now includes `gripper-1`, a `table`
frame and collision geometry whose top is world `z=0`, and the required vision
services. The first enabled cycle reached the pre-grasp waypoint, but the motion
planner blocked its grasp descent because the old code placed the gripper frame
at the object center, intersecting the modeled claws with the table. The plan
now applies the measured claw-to-frame offset.

## Security setup

The credential formerly committed to `starter-script.py` must be revoked. Do
not reuse it, even if the repository is later made private.

```bash
cp .env.example .env
```

Create a replacement machine API key in Viam and put it only in `.env`:

```dotenv
API_KEY_ID=...
API_KEY=...
```

The drop destination is a world-frame pose, not a component resource. Confirm
these values against the physical workcell before approving calibration:

```dotenv
PLACE_X_MM=300
PLACE_Y_MM=150
PLACE_Z_MM=10
GRASP_Z_OFFSET_MM=55
PLACE_Z_OFFSET_MM=55
```

The `.env` file is ignored by Git.

## Install

Python 3.11 is selected by `.python-version`.

```bash
uv sync
```

## Verification ladder

Run the unit tests without connecting to hardware:

```bash
uv run python -m unittest -v
```

Confirm the live resource and frame configuration without motion:

```bash
uv run python robot_app.py doctor
```

Detect exactly one `rectangle-green`, retrieve its 3D object geometry, and
transform the wrist-camera pose to `world`:

```bash
uv run python robot_app.py detect
```

Select a different configured color without editing `.env`:

```bash
uv run python robot_app.py detect --target-label rectangle-blue
```

Target selection accepts exact labels or shell-style patterns:

```bash
# One rectangle of any configured color
uv run python robot_app.py detect --target-label 'rectangle-*'

# One blue object of any configured shape
uv run python robot_app.py detect --target-label '*-blue'

# Exactly one detected object of any configured shape and color
uv run python robot_app.py detect --target-label '*'
```

The quotes are required so the shell does not expand `*`. Pattern selection
still fails closed when zero or multiple objects match; it never chooses
arbitrarily. The detector must already produce labels in the expected
`shape-color` form, such as `circle-red` or `triangle-blue`.

Shape/color selection does not by itself make every physical object graspable.
The current motion plan uses the detected 3D center, a fixed tool-height offset,
and a top-down grasp. Keep physical targets similarly sized and center-graspable
unless object-specific grasp offsets and orientations are calibrated.

The live shape detector uses fixed HSV thresholds. Reflections can make a block
briefly appear under a neighboring color label, so `detect` retries a bounded
number of times until the requested selector yields exactly one target. It still
fails closed if the requested target is missing or ambiguous.

### Yellow upright cylinder

The four-sided `shape-detector` cannot detect circles. The machine therefore
keeps the rectangle pipeline intact and adds a dedicated Viam
`vision/color_detector` named `yellow-cylinder-detector`. Its saved HSV settings
isolate a saturated patch on the yellow cylinder and label it `circle-yellow`.
The SDK then confirms exactly one circular top in the camera image, projects the
circle's interior through the RealSense point cloud, and infers the upright
cylinder center halfway between its observed top and `TABLE_TOP_Z_MM`.

`yellow-cylinder-segment` remains available for Viam-side 3D diagnostics, but
the physical plan uses the circle/top-depth locator because a rectangular color
bounding box can include the table or land on one side of the cylinder.

The label is a controlled alias for this workcell, not general shape
classification. If another large saturated-yellow object enters the view, the
application will fail closed unless exactly one `circle-yellow` candidate and
one nearby circular top are present.

The saved `yellow-cylinder-detector` calibration is:

```json
{
  "segment_size_px": 300,
  "hue_tolerance_pct": 0.018,
  "saturation_cutoff_pct": 0.75,
  "value_cutoff_pct": 0.2,
  "detect_color": "#FFDE00",
  "label": "circle-yellow",
  "camera_name": "camera-1"
}
```

Keep the full cylinder inside the calibrated camera view. Partial yellow
objects at an image edge are rejected before 3D planning.

Check the alternate detector and segmenter without changing `.env`:

```bash
uv run python robot_app.py doctor \
  --detector-name yellow-cylinder-detector \
  --segmenter-name yellow-cylinder-segment
```

Localize the cylinder, then print its complete no-motion plan:

```bash
uv run python robot_app.py detect \
  --target-locator circle-top \
  --detector-name yellow-cylinder-detector \
  --segmenter-name yellow-cylinder-segment \
  --target-label circle-yellow

uv run python robot_app.py pick-place \
  --target-locator circle-top \
  --detector-name yellow-cylinder-detector \
  --segmenter-name yellow-cylinder-segment \
  --target-label circle-yellow
```

Do not add `--execute` until the taller cylinder's printed object center,
gripper-frame grasp height, and jaw contact point have been physically checked.
The rectangle calibration does not prove a safe cylinder grasp.

The SDK client disables its background connection probe because transferring a
RealSense point cloud can take longer than the probe interval on this machine.
3D localization has its own `DETECTION_TIMEOUT_S` (60 seconds by default), while
ordinary resource and motion operations remain bounded by `RPC_TIMEOUT_S`.

Physical execution prints each stage immediately. If an RPC fails, the final
message identifies whether it happened during localization, approach, grasp,
transport, placement, or recovery.

Print the complete pick/place plan without sending arm or gripper commands:

```bash
uv run python robot_app.py pick-place
```

If a failed cycle leaves the wrist camera away from the calibrated viewing
pose, move only to the configured home pose before detecting again:

```bash
CALIBRATION_APPROVED=true uv run python robot_app.py home \
  --confirm-physical-motion
```

This command does not detect, grasp, or continue into a pick/place cycle.

## Physical execution gate

Before physical execution, verify these in Viam's 3D Scene:

1. `camera-1` is aligned with its depth point cloud.
2. `gripper-1` geometry matches the real tool. On this machine the modeled
   claws extend about 54.7 mm below the gripper frame, so pickup and placement
   apply a 55 mm frame offset instead of treating the object center as the TCP.
3. The table top and collision geometry align with the real table.
4. `PLACE_X_MM`, `PLACE_Y_MM`, and `PLACE_Z_MM` identify the desired drop
   location in the world frame.
5. The world-frame approach, grasp, lift, place, and retreat coordinates printed
   by the dry run are reachable and clear.
6. `GRASP_Z_OFFSET_MM` and `PLACE_Z_OFFSET_MM` account for the TCP, block height,
   and placement surface.

With the work area clear, an operator at the emergency stop, and everyone
warned, approve calibration for exactly one process and run one cycle:

```bash
CALIBRATION_APPROVED=true uv run python robot_app.py pick-place \
  --target-label 'rectangle-*' \
  --execute \
  --confirm-physical-motion
```

The application stops `arm-1` on an exception. A separate stop command is also
available:

```bash
uv run python robot_app.py stop
```

## Source guidance

- [Viam: Configure detections-to-segments](https://docs.viam.com/reference/services/vision/detections-to-segments/)
- [Viam: Pick an object](https://docs.viam.com/motion-planning/move-an-arm/pick-an-object/)
- [Viam: Place an object](https://docs.viam.com/motion-planning/move-an-arm/place-an-object/)
- [Viam: Arm with a gripper and wrist camera](https://docs.viam.com/motion-planning/frame-system/arm-gripper-camera/)
- [Viam: Motion service API](https://docs.viam.com/reference/apis/services/motion/)
