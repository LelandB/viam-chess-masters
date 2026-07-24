"""Unit tests for the no-hardware planning and safety logic."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from viam.proto.common import Pose, PoseInFrame

from robot_app import (
    ConfigurationError,
    DetectionError,
    Settings,
    WorkspaceBounds,
    build_plan,
    doctor,
    locate_target,
    select_target_geometry,
    validate_execution_request,
)


def fake_object(label: str, x: float = 300, y: float = 0, z: float = 25):
    geometry = SimpleNamespace(label=label, center=Pose(x=x, y=y, z=z))
    return SimpleNamespace(
        point_cloud=b"pcd",
        geometries=SimpleNamespace(geometries=[geometry]),
    )


def settings() -> Settings:
    return Settings(
        machine_address="example.viam.cloud",
        api_key_id="id",
        api_key="secret",
        arm_name="arm-1",
        camera_name="camera-1",
        gripper_name="gripper-1",
        detector_name="shape-detector",
        segmenter_name="vision-segment",
        motion_name="builtin",
        home_pose_name="home-pose",
        table_name="table",
        target_label="rectangle-green",
        place_x_mm=300,
        place_y_mm=150,
        place_z_mm=-10,
        detection_attempts=6,
        detection_retry_delay_s=0.5,
        approach_clearance_mm=100,
        grasp_z_offset_mm=5,
        lift_clearance_mm=150,
        place_z_offset_mm=10,
        rpc_timeout_s=30,
        settle_s=0.5,
        calibration_approved=False,
        workspace=WorkspaceBounds(100, 700, -500, 500, -50, 500),
    )


class TargetSelectionTests(unittest.TestCase):
    def test_selects_one_matching_target(self):
        target, geometry = select_target_geometry(
            [fake_object("rectangle-blue"), fake_object("rectangle-green")],
            "rectangle-green",
        )
        self.assertEqual(geometry.label, "rectangle-green")
        self.assertEqual(target.point_cloud, b"pcd")

    def test_rejects_ambiguous_targets(self):
        with self.assertRaises(DetectionError):
            select_target_geometry(
                [fake_object("rectangle-green"), fake_object("rectangle-green")],
                "rectangle-green",
            )

    def test_rejects_missing_target(self):
        with self.assertRaises(DetectionError):
            select_target_geometry([fake_object("rectangle-blue")], "rectangle-green")


class TargetLocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_retries_until_exact_target_is_available(self):
        config = replace(
            settings(),
            detection_attempts=2,
            detection_retry_delay_s=0,
        )
        vision = SimpleNamespace(
            get_object_point_clouds=AsyncMock(
                side_effect=[
                    [fake_object("rectangle-blue")],
                    [fake_object("rectangle-green")],
                ]
            )
        )
        world_pose = PoseInFrame(
            reference_frame="world",
            pose=Pose(x=300, y=0, z=25),
        )
        machine = SimpleNamespace(transform_pose=AsyncMock(return_value=world_pose))

        target = await locate_target(machine, vision, config)

        self.assertEqual(target.label, "rectangle-green")
        self.assertEqual(vision.get_object_point_clouds.await_count, 2)
        machine.transform_pose.assert_awaited_once()

    async def test_fails_closed_after_bounded_attempts(self):
        config = replace(
            settings(),
            detection_attempts=2,
            detection_retry_delay_s=0,
        )
        vision = SimpleNamespace(
            get_object_point_clouds=AsyncMock(
                return_value=[fake_object("rectangle-blue")]
            )
        )
        machine = SimpleNamespace(transform_pose=AsyncMock())

        with self.assertRaisesRegex(DetectionError, "did not stabilize"):
            await locate_target(machine, vision, config)

        self.assertEqual(vision.get_object_point_clouds.await_count, 2)
        machine.transform_pose.assert_not_awaited()


class PlanTests(unittest.TestCase):
    def test_builds_world_frame_vertical_approach_lift_and_retreat(self):
        config = settings()
        target = PoseInFrame(reference_frame="world", pose=Pose(x=300, y=-100, z=20))
        gripper = PoseInFrame(
            reference_frame="world",
            pose=Pose(o_x=0, o_y=0, o_z=-1, theta=0),
        )
        marker = PoseInFrame(reference_frame="world", pose=Pose(x=300, y=150, z=-10))
        plan = build_plan(target, gripper, marker, config)

        self.assertEqual(plan.grasp.pose.z, 25)
        self.assertEqual(plan.pre_grasp.pose.z, 125)
        self.assertEqual(plan.lift.pose.z, 175)
        self.assertEqual(plan.place.pose.z, 0)
        self.assertEqual(plan.retreat.pose.x, plan.place.pose.x)
        self.assertEqual(plan.retreat.pose.y, plan.place.pose.y)
        self.assertEqual(plan.retreat.reference_frame, "world")

    def test_rejects_plan_outside_workspace(self):
        config = settings()
        target = PoseInFrame(reference_frame="world", pose=Pose(x=900, y=0, z=20))
        origin = PoseInFrame(reference_frame="world", pose=Pose())
        with self.assertRaises(DetectionError):
            build_plan(target, origin, origin, config)


class DoctorTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_required_resource_without_frame(self):
        config = settings()
        resource_names = {
            config.arm_name,
            config.camera_name,
            config.gripper_name,
            config.detector_name,
            config.segmenter_name,
            config.home_pose_name,
            config.table_name,
        }
        machine = SimpleNamespace(
            resource_names=[SimpleNamespace(name=name) for name in resource_names],
            get_frame_system_config=AsyncMock(
                return_value=[
                    SimpleNamespace(frame=SimpleNamespace(reference_frame=frame_name))
                    for frame_name in (
                        config.arm_name,
                        config.camera_name,
                        config.gripper_name,
                    )
                ]
            ),
        )

        with self.assertRaisesRegex(ConfigurationError, "table"):
            await doctor(machine, config)


class SafetyGateTests(unittest.TestCase):
    def test_dry_run_needs_no_motion_confirmation(self):
        validate_execution_request(False, False, False)

    def test_execution_requires_confirmation_and_calibration(self):
        with self.assertRaises(ConfigurationError):
            validate_execution_request(True, False, True)
        with self.assertRaises(ConfigurationError):
            validate_execution_request(True, True, False)
        validate_execution_request(True, True, True)

    def test_environment_requires_credentials(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("robot_app.load_dotenv"),
            self.assertRaises(ConfigurationError),
        ):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()
