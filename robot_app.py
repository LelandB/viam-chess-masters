"""Safe Viam SDK app for detecting and moving one green rectangular block.

Commands are read-only unless ``pick-place --execute`` or ``stop`` is used.
Physical pick-and-place additionally requires ``--confirm-physical-motion``
and ``CALIBRATION_APPROVED=true``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from grpclib.exceptions import GRPCError, StreamTerminatedError
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient


class ConfigurationError(RuntimeError):
    """Raised when local settings are missing or unsafe."""


class DetectionError(RuntimeError):
    """Raised when a unique, usable target cannot be localized."""


class RobotMotionError(RuntimeError):
    """Raised when a requested robot action reports failure."""


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class WorkspaceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, pose: Pose) -> bool:
        return (
            self.x_min <= pose.x <= self.x_max
            and self.y_min <= pose.y <= self.y_max
            and self.z_min <= pose.z <= self.z_max
        )


@dataclass(frozen=True)
class Settings:
    machine_address: str
    api_key_id: str
    api_key: str
    arm_name: str
    camera_name: str
    gripper_name: str
    detector_name: str
    segmenter_name: str
    motion_name: str
    home_pose_name: str
    table_name: str
    target_label: str
    place_x_mm: float
    place_y_mm: float
    place_z_mm: float
    detection_attempts: int
    detection_retry_delay_s: float
    approach_clearance_mm: float
    grasp_z_offset_mm: float
    lift_clearance_mm: float
    place_z_offset_mm: float
    rpc_timeout_s: float
    settle_s: float
    calibration_approved: bool
    workspace: WorkspaceBounds

    @classmethod
    def from_env(cls) -> Settings:
        # Do not walk parent directories for credentials. This app owns only the
        # .env file beside this source file.
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
        required = ("MACHINE_ADDRESS", "API_KEY_ID", "API_KEY")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise ConfigurationError(
                "Missing environment variables: " + ", ".join(missing)
            )

        workspace = WorkspaceBounds(
            x_min=_float_env("WORKSPACE_X_MIN_MM", 100),
            x_max=_float_env("WORKSPACE_X_MAX_MM", 700),
            y_min=_float_env("WORKSPACE_Y_MIN_MM", -500),
            y_max=_float_env("WORKSPACE_Y_MAX_MM", 500),
            z_min=_float_env("WORKSPACE_Z_MIN_MM", -50),
            z_max=_float_env("WORKSPACE_Z_MAX_MM", 500),
        )
        if not (
            workspace.x_min < workspace.x_max
            and workspace.y_min < workspace.y_max
            and workspace.z_min < workspace.z_max
        ):
            raise ConfigurationError("Each workspace minimum must be below its maximum")

        approach = _float_env("APPROACH_CLEARANCE_MM", 100)
        lift = _float_env("LIFT_CLEARANCE_MM", 150)
        timeout = _float_env("RPC_TIMEOUT_S", 30)
        settle = _float_env("SETTLE_S", 0.5)
        detection_attempts = _int_env("DETECTION_ATTEMPTS", 6)
        detection_retry_delay = _float_env("DETECTION_RETRY_DELAY_S", 0.5)
        if approach <= 0 or lift <= 0 or timeout <= 0 or settle < 0:
            raise ConfigurationError(
                "Approach, lift, and timeout must be positive; settle cannot be negative"
            )
        if detection_attempts <= 0 or detection_retry_delay < 0:
            raise ConfigurationError(
                "Detection attempts must be positive; retry delay cannot be negative"
            )

        return cls(
            machine_address=os.environ["MACHINE_ADDRESS"],
            api_key_id=os.environ["API_KEY_ID"],
            api_key=os.environ["API_KEY"],
            arm_name=os.getenv("ARM_NAME", "arm-1"),
            camera_name=os.getenv("CAMERA_NAME", "camera-1"),
            gripper_name=os.getenv("GRIPPER_NAME", "gripper-1"),
            detector_name=os.getenv("DETECTOR_NAME", "shape-detector"),
            segmenter_name=os.getenv("SEGMENTER_NAME", "vision-segment"),
            motion_name=os.getenv("MOTION_NAME", "builtin"),
            home_pose_name=os.getenv("HOME_POSE_NAME", "home-pose"),
            table_name=os.getenv("TABLE_NAME", "table"),
            target_label=os.getenv("TARGET_LABEL", "rectangle-green"),
            place_x_mm=_float_env("PLACE_X_MM", 300),
            place_y_mm=_float_env("PLACE_Y_MM", 150),
            place_z_mm=_float_env("PLACE_Z_MM", -10),
            detection_attempts=detection_attempts,
            detection_retry_delay_s=detection_retry_delay,
            approach_clearance_mm=approach,
            grasp_z_offset_mm=_float_env("GRASP_Z_OFFSET_MM", 0),
            lift_clearance_mm=lift,
            place_z_offset_mm=_float_env("PLACE_Z_OFFSET_MM", 0),
            rpc_timeout_s=timeout,
            settle_s=settle,
            calibration_approved=_bool_env("CALIBRATION_APPROVED", False),
            workspace=workspace,
        )


@dataclass(frozen=True)
class Target:
    label: str
    point_cloud_bytes: int
    pose_in_camera: PoseInFrame
    pose_in_world: PoseInFrame


@dataclass(frozen=True)
class PickPlacePlan:
    pre_grasp: PoseInFrame
    grasp: PoseInFrame
    lift: PoseInFrame
    pre_place: PoseInFrame
    place: PoseInFrame
    retreat: PoseInFrame

    def steps(self) -> Iterable[tuple[str, PoseInFrame]]:
        return (
            ("pre-grasp", self.pre_grasp),
            ("grasp", self.grasp),
            ("lift", self.lift),
            ("pre-place", self.pre_place),
            ("place", self.place),
            ("retreat", self.retreat),
        )


async def connect(settings: Settings) -> RobotClient:
    options = RobotClient.Options.with_api_key(
        api_key=settings.api_key,
        api_key_id=settings.api_key_id,
    )
    return await RobotClient.at_address(settings.machine_address, options)


def _pose_at(x: float, y: float, z: float, orientation: Pose) -> Pose:
    return Pose(
        x=x,
        y=y,
        z=z,
        o_x=orientation.o_x,
        o_y=orientation.o_y,
        o_z=orientation.o_z,
        theta=orientation.theta,
    )


def _world_pose(pose: Pose) -> PoseInFrame:
    return PoseInFrame(reference_frame="world", pose=pose)


def format_pose(pose_in_frame: PoseInFrame) -> str:
    pose = pose_in_frame.pose
    return (
        f"{pose_in_frame.reference_frame}: "
        f"x={pose.x:.1f} y={pose.y:.1f} z={pose.z:.1f} mm; "
        f"orientation=({pose.o_x:.4f}, {pose.o_y:.4f}, "
        f"{pose.o_z:.4f}, {pose.theta:.2f} deg)"
    )


def select_target_geometry(
    objects: Iterable[Any], target_label: str
) -> tuple[Any, Any]:
    matches: list[tuple[Any, Any]] = []
    observed_labels: list[str] = []
    for obj in objects:
        geometries = getattr(getattr(obj, "geometries", None), "geometries", ())
        for geometry in geometries:
            label = getattr(geometry, "label", "")
            observed_labels.append(label or "<unlabeled>")
            if label == target_label:
                matches.append((obj, geometry))

    if len(matches) != 1:
        labels = ", ".join(observed_labels) if observed_labels else "none"
        raise DetectionError(
            f"Expected exactly one {target_label!r}; found {len(matches)}. "
            f"Observed labels: {labels}"
        )
    return matches[0]


def require_in_workspace(name: str, pose: Pose, bounds: WorkspaceBounds) -> None:
    if not bounds.contains(pose):
        raise DetectionError(
            f"{name} is outside configured workspace bounds: "
            f"x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}"
        )


async def locate_target(
    machine: RobotClient,
    vision: VisionClient,
    settings: Settings,
) -> Target:
    last_error: DetectionError | None = None
    for attempt in range(1, settings.detection_attempts + 1):
        objects = await vision.get_object_point_clouds(
            settings.camera_name,
            timeout=settings.rpc_timeout_s,
        )
        try:
            obj, geometry = select_target_geometry(objects, settings.target_label)
            break
        except DetectionError as exc:
            last_error = exc
            if attempt == settings.detection_attempts:
                raise DetectionError(
                    f"Target did not stabilize after {settings.detection_attempts} "
                    f"attempts. Last result: {exc}"
                ) from exc
            print(
                f"Detection attempt {attempt}/{settings.detection_attempts} "
                f"did not yield one {settings.target_label!r}; retrying"
            )
            await asyncio.sleep(settings.detection_retry_delay_s)
    else:  # pragma: no cover - defensive; the loop always returns or raises.
        raise DetectionError(str(last_error))

    pose_in_camera = PoseInFrame(
        reference_frame=settings.camera_name,
        pose=geometry.center,
    )
    # Freeze the target before moving. The wrist camera frame moves with the arm.
    pose_in_world = await machine.transform_pose(pose_in_camera, "world")
    require_in_workspace("Detected target", pose_in_world.pose, settings.workspace)
    return Target(
        label=geometry.label,
        point_cloud_bytes=len(getattr(obj, "point_cloud", b"")),
        pose_in_camera=pose_in_camera,
        pose_in_world=pose_in_world,
    )


def build_plan(
    target_world: PoseInFrame,
    gripper_world: PoseInFrame,
    place_marker_world: PoseInFrame,
    settings: Settings,
) -> PickPlacePlan:
    target = target_world.pose
    marker = place_marker_world.pose
    orientation = gripper_world.pose
    grasp_z = target.z + settings.grasp_z_offset_mm
    place_z = marker.z + settings.place_z_offset_mm

    poses = {
        "pre_grasp": _world_pose(
            _pose_at(
                target.x,
                target.y,
                grasp_z + settings.approach_clearance_mm,
                orientation,
            )
        ),
        "grasp": _world_pose(_pose_at(target.x, target.y, grasp_z, orientation)),
        "lift": _world_pose(
            _pose_at(
                target.x,
                target.y,
                grasp_z + settings.lift_clearance_mm,
                orientation,
            )
        ),
        "pre_place": _world_pose(
            _pose_at(
                marker.x,
                marker.y,
                place_z + settings.approach_clearance_mm,
                orientation,
            )
        ),
        "place": _world_pose(_pose_at(marker.x, marker.y, place_z, orientation)),
        "retreat": _world_pose(
            _pose_at(
                marker.x,
                marker.y,
                place_z + settings.approach_clearance_mm,
                orientation,
            )
        ),
    }
    for name, pose in poses.items():
        require_in_workspace(name.replace("_", " "), pose.pose, settings.workspace)
    return PickPlacePlan(**poses)


def validate_execution_request(
    execute: bool,
    confirmed_physical_motion: bool,
    calibration_approved: bool,
) -> None:
    if not execute:
        return
    if not confirmed_physical_motion:
        raise ConfigurationError("Physical motion requires --confirm-physical-motion")
    if not calibration_approved:
        raise ConfigurationError(
            "Physical motion is blocked until CALIBRATION_APPROVED=true"
        )


async def move_checked(
    motion: MotionClient,
    component_name: str,
    destination: PoseInFrame,
    step_name: str,
    timeout: float,
    *,
    linear: bool = False,
) -> None:
    constraints = None
    if linear:
        constraints = Constraints(
            linear_constraint=[LinearConstraint(line_tolerance_mm=5.0)]
        )
    moved = await motion.move(
        component_name=component_name,
        destination=destination,
        constraints=constraints,
        timeout=timeout,
    )
    if not moved:
        raise RobotMotionError(f"Motion service reported failure during {step_name}")


async def doctor(machine: RobotClient, settings: Settings) -> None:
    resources = sorted(str(resource.name) for resource in machine.resource_names)
    required = {
        settings.arm_name,
        settings.camera_name,
        settings.gripper_name,
        settings.detector_name,
        settings.segmenter_name,
        settings.home_pose_name,
        settings.table_name,
    }
    missing = sorted(required.difference(resources))
    print(f"Connected to {settings.machine_address} ({len(resources)} resources)")
    if missing:
        raise ConfigurationError("Missing live resources: " + ", ".join(missing))

    frame_configs = await machine.get_frame_system_config()
    frames = {config.frame.reference_frame for config in frame_configs}
    required_frames = {
        settings.arm_name,
        settings.camera_name,
        settings.gripper_name,
        settings.table_name,
    }
    missing_frames = sorted(required_frames.difference(frames))
    if missing_frames:
        raise ConfigurationError(
            "Resources missing from the live frame system: "
            + ", ".join(missing_frames)
            + ". Add a Frame configuration that connects each one to world."
        )

    camera = Camera.from_robot(machine, settings.camera_name)
    detector = VisionClient.from_robot(machine, settings.detector_name)
    segmenter = VisionClient.from_robot(machine, settings.segmenter_name)
    gripper = Gripper.from_robot(machine, settings.gripper_name)
    home = Switch.from_robot(machine, settings.home_pose_name)
    motion = MotionClient.from_robot(machine, settings.motion_name)

    camera_properties = await camera.get_properties(timeout=settings.rpc_timeout_s)
    detector_properties = await detector.get_properties(timeout=settings.rpc_timeout_s)
    segmenter_properties = await segmenter.get_properties(
        timeout=settings.rpc_timeout_s
    )
    home_position = await home.get_position(timeout=settings.rpc_timeout_s)
    gripper_moving = await gripper.is_moving(timeout=settings.rpc_timeout_s)
    holding = await gripper.is_holding_something(timeout=settings.rpc_timeout_s)
    gripper_world = await motion.get_pose(
        settings.gripper_name,
        "world",
        timeout=settings.rpc_timeout_s,
    )
    place_world = _world_pose(
        Pose(
            x=settings.place_x_mm,
            y=settings.place_y_mm,
            z=settings.place_z_mm,
        )
    )

    print(f"camera supports point clouds: {camera_properties.supports_pcd}")
    print(f"detector properties: {detector_properties}")
    print(f"segmenter properties: {segmenter_properties}")
    print(f"home switch position: {home_position}")
    print(f"gripper moving: {gripper_moving}; holding status: {holding}")
    print(f"gripper pose: {format_pose(gripper_world)}")
    print(f"configured place pose: {format_pose(place_world)}")
    if not camera_properties.supports_pcd:
        raise ConfigurationError("camera-1 must support point clouds")
    print("Doctor checks passed")


async def detect(machine: RobotClient, settings: Settings) -> Target:
    vision = VisionClient.from_robot(machine, settings.segmenter_name)
    target = await locate_target(machine, vision, settings)
    print(
        f"Detected {target.label!r}; point-cloud payload "
        f"{target.point_cloud_bytes} bytes"
    )
    print(f"camera pose: {format_pose(target.pose_in_camera)}")
    print(f"world pose:  {format_pose(target.pose_in_world)}")
    return target


async def pick_place(
    machine: RobotClient,
    settings: Settings,
    *,
    execute: bool,
    confirmed_physical_motion: bool,
) -> None:
    validate_execution_request(
        execute,
        confirmed_physical_motion,
        settings.calibration_approved,
    )
    arm = Arm.from_robot(machine, settings.arm_name)
    gripper = Gripper.from_robot(machine, settings.gripper_name)
    home = Switch.from_robot(machine, settings.home_pose_name)
    motion = MotionClient.from_robot(machine, settings.motion_name)
    vision = VisionClient.from_robot(machine, settings.segmenter_name)

    try:
        if execute:
            holding_before = await gripper.is_holding_something(
                timeout=settings.rpc_timeout_s
            )
            if holding_before is True:
                raise RobotMotionError(
                    "Gripper already reports holding an object; clear it before starting"
                )
            await home.set_position(2, timeout=settings.rpc_timeout_s)
        else:
            print(
                "DRY RUN: the arm was not moved to home; detection uses its current pose"
            )

        target = await locate_target(machine, vision, settings)
        gripper_world = await motion.get_pose(
            settings.gripper_name,
            "world",
            timeout=settings.rpc_timeout_s,
        )
        place_world = _world_pose(
            Pose(
                x=settings.place_x_mm,
                y=settings.place_y_mm,
                z=settings.place_z_mm,
            )
        )
        require_in_workspace("Place pose", place_world.pose, settings.workspace)
        plan = build_plan(target.pose_in_world, gripper_world, place_world, settings)

        print(f"Target: {target.label!r}")
        for name, pose in plan.steps():
            print(f"- {name}: {format_pose(pose)}")
        if not execute:
            print("DRY RUN COMPLETE: no arm or gripper commands were sent")
            return

        await move_checked(
            motion,
            settings.gripper_name,
            plan.pre_grasp,
            "pre-grasp approach",
            settings.rpc_timeout_s,
        )
        await gripper.open(timeout=settings.rpc_timeout_s)
        await asyncio.sleep(settings.settle_s)
        await move_checked(
            motion,
            settings.gripper_name,
            plan.grasp,
            "linear grasp descent",
            settings.rpc_timeout_s,
            linear=True,
        )
        grabbed = await gripper.grab(timeout=settings.rpc_timeout_s)
        if not grabbed:
            raise RobotMotionError("Gripper closed without confirming a grasp")
        await asyncio.sleep(settings.settle_s)
        holding = await gripper.is_holding_something(timeout=settings.rpc_timeout_s)
        print(f"gripper holding status: {holding}")
        if holding is False:
            raise RobotMotionError("Gripper does not report holding the block")
        await move_checked(
            motion,
            settings.gripper_name,
            plan.lift,
            "vertical lift",
            settings.rpc_timeout_s,
            linear=True,
        )
        await move_checked(
            motion,
            settings.gripper_name,
            plan.pre_place,
            "transport to pre-place",
            settings.rpc_timeout_s,
        )
        await move_checked(
            motion,
            settings.gripper_name,
            plan.place,
            "linear place descent",
            settings.rpc_timeout_s,
            linear=True,
        )
        await gripper.open(timeout=settings.rpc_timeout_s)
        await asyncio.sleep(settings.settle_s)
        await move_checked(
            motion,
            settings.gripper_name,
            plan.retreat,
            "vertical retreat",
            settings.rpc_timeout_s,
            linear=True,
        )
        await home.set_position(2, timeout=settings.rpc_timeout_s)
        print("Pick-and-place cycle complete")
    except Exception:
        await arm.stop(timeout=settings.rpc_timeout_s)
        raise


async def stop(machine: RobotClient, settings: Settings) -> None:
    await Arm.from_robot(machine, settings.arm_name).stop(
        timeout=settings.rpc_timeout_s
    )
    print(f"Stop sent to {settings.arm_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Run read-only resource and frame checks")
    subparsers.add_parser("detect", help="Locate one target in camera and world frames")

    pick = subparsers.add_parser(
        "pick-place",
        help="Plan one bounded cycle; defaults to a no-motion dry run",
    )
    pick.add_argument(
        "--execute",
        action="store_true",
        help="Send the planned commands to physical hardware",
    )
    pick.add_argument(
        "--confirm-physical-motion",
        action="store_true",
        help="Acknowledge that the command can move the real arm and gripper",
    )
    subparsers.add_parser("stop", help="Immediately request that arm-1 stop")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    machine = await connect(settings)
    try:
        if args.command == "doctor":
            await doctor(machine, settings)
        elif args.command == "detect":
            await detect(machine, settings)
        elif args.command == "pick-place":
            await pick_place(
                machine,
                settings,
                execute=args.execute,
                confirmed_physical_motion=args.confirm_physical_motion,
            )
        elif args.command == "stop":
            await stop(machine, settings)
    finally:
        await machine.close()


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(async_main(args))
    except (ConfigurationError, DetectionError, RobotMotionError) as exc:
        raise SystemExit(f"Blocked: {exc}") from None
    except (GRPCError, StreamTerminatedError, TimeoutError) as exc:
        raise SystemExit(f"Blocked: Viam SDK connection or RPC failed: {exc}") from None


if __name__ == "__main__":
    main()
