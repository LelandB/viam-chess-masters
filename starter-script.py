# """
# Viam Pick-and-Place Workshop — Starter Script
# ==============================================

# Copy your connection details from the machine's Connect tab -> Python SDK.
# Fill in the TODOs in order. Run after each section to verify before moving on.

#     uv run python starter-script.py        # recommended (uv reads ../scripts/pyproject.toml)
#     # or, without uv:
#     python3 starter-script.py

# Prerequisites:
#     python3 --version                                # must be 3.10+
#     uv add viam-sdk                                  # or: pip install viam-sdk
#     uv run python -c "import viam; print(viam.__version__)"
# """

# import asyncio

# from viam.robot.client import RobotClient
# from viam.components.gripper import Gripper
# from viam.components.switch import Switch
# from viam.services.motion import MotionClient
# from viam.services.vision import VisionClient
# from viam.proto.common import PoseInFrame, Pose

import asyncio

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.switch import Switch
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService
from viam.services.mlmodel import MLModelClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.proto.common import PoseInFrame, Pose

# # --- Tuning constants ---------------------------------------------------------
GRIPPER_LENGTH_MM = (
    -60
)  # offset from the gripper's claw-geometry TCP to the real fingertip contact point
APPROACH_MM = -100  # clearance above the block top before descending

# # --- TODO 1: paste address + API key from the Connect tab ---------------------
MACHINE_ADDRESS = "robot12-main.ag9khwy6jn.viam.cloud"
API_KEY = "2j803k895dgbss49p155ko5yb2z4gd4b"
API_KEY_ID = "cf3b272c-ca3a-4622-812c-3a58500822cc"

# # --- Resource names (must match the CONFIGURE tab exactly) --------------------
ARM_NAME = "arm-1"
GRIPPER_NAME = "gripper-1"
CAMERA_NAME = "camera-1"
# VISION_NAME = "vision-segment"
HOME_POSE = "home-pose"
APPROACH_POSE = "approach-pose"
GRASP_POSE = "grasp-pose"
TRAVEL_POSE = "travel-pose"
PLACE_POSE = "place-pose"
VISION_NAME = "vision-segment"


async def connect():
    opts = RobotClient.Options.with_api_key(
         
        api_key=API_KEY,
        
        api_key_id=API_KEY_ID
    )
    
    return await RobotClient.at_address(MACHINE_ADDRESS, opts)

# async def main():
#     async with await connect() as machine:
#         print('Resources:')
#         print(machine.resource_names)
        
#         # arm-1
#         arm_1 = Arm.from_robot(machine, "arm-1")
#         arm_1_return_value = await arm_1.get_end_position()
#         print(f"arm-1 get_end_position return value: {arm_1_return_value}")

#         # camera-1
#         camera_1 = Camera.from_robot(machine, "camera-1")
#         camera_1_return_value = await camera_1.get_images()
#         print(f"camera-1 get_images return value: {camera_1_return_value}")

#         # home-pose
#         home_pose = Switch.from_robot(machine, "home-pose")
#         home_pose_return_value = await home_pose.get_position()
#         print(f"home-pose get_position return value: {home_pose_return_value}")

#         # approach-pose
#         approach_pose = Switch.from_robot(machine, "approach-pose")
#         approach_pose_return_value = await approach_pose.get_position()
#         print(f"approach-pose get_position return value: {approach_pose_return_value}")

#         # grasp-pose
#         grasp_pose = Switch.from_robot(machine, "grasp-pose")
#         grasp_pose_return_value = await grasp_pose.get_position()
#         print(f"grasp-pose get_position return value: {grasp_pose_return_value}")

#         # travel-pose
#         travel_pose = Switch.from_robot(machine, "travel-pose")
#         travel_pose_return_value = await travel_pose.get_position()
#         print(f"travel-pose get_position return value: {travel_pose_return_value}")

#         # place-pose
#         place_pose = Switch.from_robot(machine, "place-pose")
#         place_pose_return_value = await place_pose.get_position()
#         print(f"place-pose get_position return value: {place_pose_return_value}")

#         # gripper-1
#         gripper_1 = Gripper.from_robot(machine, "gripper-1")
#         gripper_1_return_value = await gripper_1.is_moving()
#         print(f"gripper-1 is_moving return value: {gripper_1_return_value}")

#         # table
#         table = Gripper.from_robot(machine, "table")
#         table_return_value = await table.is_moving()
#         print(f"table is_moving return value: {table_return_value}")

#         # pick-marker
#         pick_marker = Gripper.from_robot(machine, "pick-marker")
#         pick_marker_return_value = await pick_marker.is_moving()
#         print(f"pick-marker is_moving return value: {pick_marker_return_value}")

#         # place-marker
#         place_marker = Gripper.from_robot(machine, "place-marker")
#         place_marker_return_value = await place_marker.is_moving()
#         print(f"place-marker is_moving return value: {place_marker_return_value}")

        # Note that the following block is commented out because it may actuate
        # or because its argument semantics are unknown. Use with caution.
        # code-1
        # code_1 = GenericService.from_robot(machine, "code-1")
        # code_1_return_value = await code_1.do_command({})
        # print(f"code-1 do_command return value: {code_1_return_value}")

        # mlmodel-1
        # mlmodel_1 = MLModelClient.from_robot(machine, "mlmodel-1")
        # mlmodel_1_return_value = await mlmodel_1.metadata()
        # print(f"mlmodel-1 metadata return value: {mlmodel_1_return_value}")

        # # detector
        # detector = VisionClient.from_robot(machine, "detector")
        # detector_return_value = await detector.get_properties()
        # print(f"detector get_properties return value: {detector_return_value}")

        # # mlmodel-2
        # mlmodel_2 = MLModelClient.from_robot(machine, "mlmodel-2")
        # mlmodel_2_return_value = await mlmodel_2.metadata()
        # print(f"mlmodel-2 metadata return value: {mlmodel_2_return_value}")

        # # vision-1
        # vision_1 = VisionClient.from_robot(machine, "vision-1")
        # vision_1_return_value = await vision_1.get_properties()
        # print(f"vision-1 get_properties return value: {vision_1_return_value}")

# if __name__ == '__main__':
#     asyncio.run(main())

def offset_pose(pose: Pose, z_offset_mm: float) -> Pose:
    """Raise or lower a pose in z while keeping x/y/orientation fixed."""
    return Pose(
        x=pose.x,
        y=pose.y,
        z=pose.z + z_offset_mm,
        o_x=pose.o_x,
        o_y=pose.o_y,
        o_z=pose.o_z,
        theta=pose.theta,
    )

async def connect() -> RobotClient:
    opts = RobotClient.Options.with_api_key(
        api_key=API_KEY,
        api_key_id=API_KEY_ID,
    )
    return await RobotClient.at_address(MACHINE_ADDRESS, opts)


async def main() -> None:
    async with await connect() as machine:
        # TODO 2: confirm the connection — list every resource on the machine.
        # You should see arm-1, gripper-1, cam-1, the poses as Switches,
        # and the obstacles as grippers.
        print(machine.resource_names)

        # TODO 3: get typed resource handles.
        gripper = Gripper.from_robot(machine, GRIPPER_NAME)

        home = Switch.from_robot(machine, HOME_POSE)
        approach = Switch.from_robot(machine, APPROACH_POSE)
        grasp = Switch.from_robot(machine, GRASP_POSE)
        travel = Switch.from_robot(machine, TRAVEL_POSE)
        place_pose = Switch.from_robot(machine, PLACE_POSE)

        # Used in Phase 5
        motion = MotionClient.from_robot(machine, "builtin")
        vision = VisionClient.from_robot(machine, VISION_NAME)

        # TODO 4: run the static sequence (Phase 4.4).
        # Same order you tested manually from the Control tab in Phase 3.

        # SetPosition(2) executes a saved pose.
        # await home.set_position(2)
        # await approach.set_position(2)
        # await gripper.open()
        # await grasp.set_position(2)
        # await gripper.grab()
        # await asyncio.sleep(0.3)  # finger gripper settle
        # await travel.set_position(2)
        # await place_pose.set_position(2)
        # await gripper.open()
        # await home.set_position(2)
        # print("Static sequence complete")

        # TODO 5: add perception (Phase 5.5).
        # Uncomment and complete. Must be at home before detecting because the
        # camera is wrist-mounted — its frame moves with the arm.
        
        await home.set_position(2)
        objects = await vision.get_object_point_clouds(CAMERA_NAME)
        if not objects:
            print("No objects detected")
            return
        obj = max(objects, key=lambda o: len(o.point_cloud))
        label = obj.geometries.geometries[0].label
        print(f"Detected: {label}")
        
        # Create the object pose in the camera frame
        obj_in_cam = PoseInFrame(
            reference_frame=CAMERA_NAME,
            pose=obj.geometries.geometries[0].center,
        )

        # TODO 6: compute the approach and grasp poses (Phase 5.6).
        # The approach pose is worked for you — a clearance standoff above the block:
        approach_pose = offset_pose(obj_in_cam.pose, APPROACH_MM)
        #
        # Now YOU compute the grasp pose. motion.move drives the gripper-1 frame
        # (the gripper's TCP, already offset down the arm) to the target — so the
        # grasp offset is the gripper-TCP-to-fingertip depth (GRIPPER_LENGTH_MM),
        # not the whole arm reach. Fill in the offset:
        grasp_pose = offset_pose(obj_in_cam.pose, GRIPPER_LENGTH_MM)   # TODO: your offset

        # TODO 7: run the full perception-guided pick loop (Phase 5.6).
        # Hybrid approach: motion.move for the pick (Cartesian precision),
        # arm-position-saver switches for the place (pre-measured, reliable).
        #
        await motion.move("gripper-1", PoseInFrame(reference_frame=CAMERA_NAME, pose=approach_pose))
        await gripper.open()
        await motion.move("gripper-1", PoseInFrame(reference_frame=CAMERA_NAME, pose=grasp_pose))
        await gripper.grab()
        await asyncio.sleep(0.3)
        await travel.set_position(2)
        await place_pose.set_position(2)
        await gripper.open()
        await home.set_position(2)


if __name__ == "__main__":
    asyncio.run(main())
