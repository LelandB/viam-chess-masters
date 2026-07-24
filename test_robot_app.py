"""Unit tests for the no-hardware planning and safety logic."""

import os
import struct
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from viam.proto.common import Pose, PoseInFrame

from robot_app import (
    ConfigurationError,
    DetectionError,
    ImageCircle,
    Settings,
    WorkspaceBounds,
    build_parser,
    build_plan,
    camera_point_at_circle,
    connect,
    cylinder_center_from_top,
    doctor,
    find_target_circle,
    locate_target,
    move_home,
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
        api_key_id="00000000-0000-0000-0000-000000000001",
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
        target_locator="segments",
        table_top_z_mm=0,
        place_x_mm=300,
        place_y_mm=150,
        place_z_mm=10,
        detection_attempts=6,
        detection_retry_delay_s=0.5,
        detection_timeout_s=60,
        approach_clearance_mm=100,
        grasp_z_offset_mm=55,
        lift_clearance_mm=150,
        place_z_offset_mm=55,
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

    def test_selects_one_rectangle_of_any_color(self):
        target, geometry = select_target_geometry(
            [fake_object("circle-yellow"), fake_object("rectangle-blue")],
            "rectangle-*",
        )
        self.assertEqual(geometry.label, "rectangle-blue")
        self.assertEqual(target.point_cloud, b"pcd")

    def test_any_color_rejects_multiple_rectangles(self):
        with self.assertRaises(DetectionError):
            select_target_geometry(
                [fake_object("rectangle-blue"), fake_object("rectangle-green")],
                "rectangle-*",
            )

    def test_selects_one_shape_of_requested_color(self):
        _, geometry = select_target_geometry(
            [fake_object("circle-red"), fake_object("triangle-blue")],
            "*-blue",
        )
        self.assertEqual(geometry.label, "triangle-blue")

    def test_selects_one_object_of_any_shape_and_color(self):
        _, geometry = select_target_geometry(
            [fake_object("triangle-yellow")],
            "*",
        )
        self.assertEqual(geometry.label, "triangle-yellow")

    def test_any_shape_and_color_rejects_multiple_objects(self):
        with self.assertRaises(DetectionError):
            select_target_geometry(
                [fake_object("circle-red"), fake_object("triangle-blue")],
                "*",
            )

    def test_any_shape_and_color_ignores_unlabeled_geometry(self):
        _, geometry = select_target_geometry(
            [fake_object(""), fake_object("triangle-blue")],
            "*",
        )
        self.assertEqual(geometry.label, "triangle-blue")


class CommandLineTests(unittest.TestCase):
    def test_detect_accepts_alternate_segmenter_and_label(self):
        args = build_parser().parse_args(
            [
                "detect",
                "--detector-name",
                "yellow-cylinder-detector",
                "--segmenter-name",
                "yellow-cylinder-segment",
                "--target-label",
                "circle-yellow",
                "--target-locator",
                "circle-top",
            ]
        )

        self.assertEqual(args.segmenter_name, "yellow-cylinder-segment")
        self.assertEqual(args.detector_name, "yellow-cylinder-detector")
        self.assertEqual(args.target_label, "circle-yellow")
        self.assertEqual(args.target_locator, "circle-top")

    def test_doctor_accepts_alternate_vision_pipeline(self):
        args = build_parser().parse_args(
            [
                "doctor",
                "--detector-name",
                "yellow-cylinder-detector",
                "--segmenter-name",
                "yellow-cylinder-segment",
            ]
        )

        self.assertEqual(args.detector_name, "yellow-cylinder-detector")
        self.assertEqual(args.segmenter_name, "yellow-cylinder-segment")


class CircleLocalizationTests(unittest.TestCase):
    def test_finds_one_circle_near_color_candidate(self):
        image = np.full((480, 640, 3), 35, dtype=np.uint8)
        cv2.circle(image, (400, 250), 45, (0, 220, 255), -1)
        cv2.circle(image, (400, 250), 45, (10, 10, 10), 3)
        encoded, jpeg = cv2.imencode(".jpg", image)
        self.assertTrue(encoded)
        detection = SimpleNamespace(x_min=345, x_max=365, y_min=240, y_max=270)

        circle = find_target_circle(jpeg.tobytes(), detection)

        self.assertAlmostEqual(circle.x_px, 400, delta=3)
        self.assertAlmostEqual(circle.y_px, 250, delta=3)
        self.assertAlmostEqual(circle.radius_px, 45, delta=4)

    def test_projects_circle_depth_points_to_camera_millimeters(self):
        header = (
            b"VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
            b"TYPE F F F I\nCOUNT 1 1 1 1\nWIDTH 40\nHEIGHT 1\n"
            b"POINTS 40\nDATA binary\n"
        )
        payload = header + b"".join(
            struct.pack("<fffI", 0.1, 0.2, 1.0, 0xFFFF00) for _ in range(40)
        )
        intrinsics = SimpleNamespace(
            focal_x_px=100,
            focal_y_px=100,
            center_x_px=0,
            center_y_px=0,
        )

        pose = camera_point_at_circle(
            payload,
            intrinsics,
            ImageCircle(x_px=10, y_px=20, radius_px=10),
        )

        self.assertAlmostEqual(pose.x, 100, places=3)
        self.assertAlmostEqual(pose.y, 200, places=3)
        self.assertAlmostEqual(pose.z, 1000, places=3)

    def test_infers_cylinder_center_between_table_and_top(self):
        center, height = cylinder_center_from_top(
            PoseInFrame(reference_frame="world", pose=Pose(x=400, y=-50, z=40)),
            table_top_z_mm=0,
        )

        self.assertEqual(height, 40)
        self.assertEqual(center.pose.x, 400)
        self.assertEqual(center.pose.y, -50)
        self.assertEqual(center.pose.z, 20)


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_disables_background_probe_for_large_point_cloud_rpc(self):
        with patch("robot_app.RobotClient.at_address", new=AsyncMock()) as at_address:
            await connect(settings())

        options = at_address.await_args.args[1]
        self.assertEqual(options.check_connection_interval, 0)
        self.assertEqual(options.attempt_reconnect_interval, 0)


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
        vision.get_object_point_clouds.assert_awaited_with(
            config.camera_name,
            timeout=config.detection_timeout_s,
        )
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
        marker = PoseInFrame(reference_frame="world", pose=Pose(x=300, y=150, z=10))
        plan = build_plan(target, gripper, marker, config)

        self.assertEqual(plan.grasp.pose.z, 75)
        self.assertEqual(plan.pre_grasp.pose.z, 175)
        self.assertEqual(plan.lift.pose.z, 225)
        self.assertEqual(plan.place.pose.z, 65)
        self.assertEqual(plan.pre_place.pose.z, 165)
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


class HomeCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_home_requires_confirmation_before_resolving_resources(self):
        machine = SimpleNamespace()

        with self.assertRaises(ConfigurationError):
            await move_home(
                machine,
                settings(),
                confirmed_physical_motion=False,
            )


if __name__ == "__main__":
    unittest.main()
