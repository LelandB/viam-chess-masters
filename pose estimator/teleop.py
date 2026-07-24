import asyncio
import cv2
import time
import mediapipe as mp
import numpy as np
import math

from viam.robot.client import RobotClient
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose

# ---------------------------------------------------------
# MEDIAPIPE & OPENCV SETUP
# ---------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        
    (0, 5), (5, 6), (6, 7), (7, 8),        
    (5, 9), (9, 10), (10, 11), (11, 12),   
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) 
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='mediapipe/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    num_hands=1
)

class SimpleEMA:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def update(self, current_val):
        if self.value is None:
            self.value = current_val
        else:
            self.value = self.alpha * current_val + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None

def get_orthogonal_axes(p0, p5, p9, w, h):
    v1 = np.array([(p5.x - p0.x) * w, (p5.y - p0.y) * h, (p5.z - p0.z) * w])
    v2 = np.array([(p9.x - p0.x) * w, (p9.y - p0.y) * h, (p9.z - p0.z) * w])
    
    x_axis = v1 / (np.linalg.norm(v1) + 1e-6)
    z_axis = np.cross(v1, v2)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
    
    return x_axis, y_axis, z_axis

def draw_axes(frame, origin, axes, w, h, length=60):
    x_axis, y_axis, z_axis = axes
    o_x, o_y = int(origin.x * w), int(origin.y * h)
    
    pt_x = (int(o_x + x_axis[0] * length), int(o_y + x_axis[1] * length))
    pt_y = (int(o_x + y_axis[0] * length), int(o_y + y_axis[1] * length))
    pt_z = (int(o_x + z_axis[0] * length), int(o_y + z_axis[1] * length))

    cv2.line(frame, (o_x, o_y), pt_x, (0, 0, 255), 3) 
    cv2.line(frame, (o_x, o_y), pt_y, (0, 255, 0), 3) 
    cv2.line(frame, (o_x, o_y), pt_z, (255, 0, 0), 3) 

# ---------------------------------------------------------
# VIAM CONNECTION LOGIC
# ---------------------------------------------------------
async def connect_to_viam():
    ROBOT_ADDRESS = "robot12-main.ag9khwy6jn.viam.cloud"
    API_KEY_ID = "578c98f8-53e4-4827-bf1a-0a2c87e09681"
    API_KEY = "z2jj7tak3cdzojakzls69akqb97i7sxm"
    
    opts = RobotClient.Options.with_api_key(
        api_key_id=API_KEY_ID,
        api_key=API_KEY
    )
    
    print(f"Attempting to connect to robot at {ROBOT_ADDRESS}...")
    try:
        robot = await RobotClient.at_address(ROBOT_ADDRESS, opts)
        print("Connected successfully!")
        return robot
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None

# ---------------------------------------------------------
# MAIN ASYNC LOOP
# ---------------------------------------------------------
async def main():
    robot = await connect_to_viam()
    if not robot:
        return
        
    arm = Arm.from_robot(robot, "arm-1")
    gripper = Gripper.from_robot(robot, "gripper-1")
    
    # Viam Teleop variables
    arm_baseline_pose = None
    last_command_time = time.time()
    COMMAND_RATE_LIMIT = 0.15  
    
    # --- TUNING VARIABLES ---
    ENABLE_ROTATION = False    # FLAG: Toggle hand rotation tracking on/off
    REAL_WORLD_SCALE = 3.0     
    MOVEMENT_VELOCITY = 5      
    MOVE_THRESHOLD_MM = 3.0
    last_commanded_xyz = None
    
    # State Trackers
    hand_in_frame_last_tick = False
    is_gripping = False
    movement_paused = False    
    last_space_press_time = 0.0  

    cap = cv2.VideoCapture(0)
    window_name = 'Pose Estimator'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    SCALE_X, SCALE_Y, SCALE_Z = 1.0, 1.0, 1.0 
    PINCH_MIN_RATIO, PINCH_MAX_RATIO = 0.15, 1.20  

    baseline_wrist = None
    baseline_axes = None
    baseline_palm_size = None

    # EMA Filters
    ema_dx = SimpleEMA(alpha=0.25)
    ema_dy = SimpleEMA(alpha=0.25)
    ema_dz = SimpleEMA(alpha=0.25)
    ema_pinch = SimpleEMA(alpha=0.25) 
    ema_rx = SimpleEMA(alpha=0.25)
    ema_ry = SimpleEMA(alpha=0.25)
    ema_rz = SimpleEMA(alpha=0.25)

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
    
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
    
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)
    
            key = cv2.waitKey(1) & 0xFF
            current_time_loop = time.time()
            
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('r'):
                current_result = latest_result 
                if current_result and current_result.hand_landmarks and len(current_result.hand_landmarks) > 0:
                    
                    # 1. Reset Camera Baseline
                    landmarks = current_result.hand_landmarks[0]
                    baseline_wrist = landmarks[0]
                    baseline_p5 = landmarks[5]
                    
                    baseline_axes = get_orthogonal_axes(baseline_wrist, baseline_p5, landmarks[9], w, h)
                    baseline_palm_size = np.sqrt(((baseline_p5.x - baseline_wrist.x)*w)**2 + ((baseline_p5.y - baseline_wrist.y)*h)**2)
                    
                    ema_dx.reset()
                    ema_dy.reset()
                    ema_dz.reset()
                    ema_pinch.reset()
                    ema_rx.reset()
                    ema_ry.reset()
                    ema_rz.reset()
                    
                    # 2. Reset Robot Baseline
                    print("Retrieving current arm pose for baseline...")
                    arm_baseline_pose = await arm.get_end_position()
                    last_commanded_xyz = None 
                    print(f"Robot baseline set at: X:{arm_baseline_pose.x:.1f}, Y:{arm_baseline_pose.y:.1f}, Z:{arm_baseline_pose.z:.1f}")
            
            # --- SPACEBAR TOGGLE (With Debounce) ---
            elif key == 32: 
                if (current_time_loop - last_space_press_time) > 0.3:
                    movement_paused = not movement_paused
                    print(f"Movement Paused: {movement_paused}")
                    last_space_press_time = current_time_loop
                    if movement_paused:
                        asyncio.create_task(arm.stop())
                    
            # --- DRAW STATUS OVERLAY ---
            status_text = "PAUSED (Press Space)" if movement_paused else "ACTIVE (Press Space)"
            status_color = (0, 0, 255) if movement_paused else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
            current_result = latest_result 
            
            # --- OUT OF FRAME HANDLING ---
            if not current_result or not current_result.hand_landmarks or len(current_result.hand_landmarks) == 0:
                if hand_in_frame_last_tick and not movement_paused:
                    print("Hand lost. Stopping arm.")
                    asyncio.create_task(arm.stop())
                    hand_in_frame_last_tick = False
                    
                ema_dx.reset()
                ema_dy.reset()
                ema_dz.reset()
                ema_pinch.reset()
                ema_rx.reset()
                ema_ry.reset()
                ema_rz.reset()
            else:
                hand_in_frame_last_tick = True
                hand_landmarks = current_result.hand_landmarks[0]
                
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]

                    start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                    end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

                current_wrist = hand_landmarks[0]
                p5 = hand_landmarks[5]
                p9 = hand_landmarks[9]
                p4 = hand_landmarks[4]  
                p8 = hand_landmarks[8]  
                
                # Pinch Calculation
                raw_pinch_dist = np.sqrt((p8.x - p4.x)**2 + (p8.y - p4.y)**2 + (p8.z - p4.z)**2)
                current_palm_size = np.sqrt(((p5.x - current_wrist.x)*w)**2 + ((p5.y - current_wrist.y)*h)**2)
                raw_pinch_ratio = raw_pinch_dist / ((current_palm_size / w) + 1e-6)
                raw_pinch_variable = np.clip((raw_pinch_ratio - PINCH_MIN_RATIO) / (PINCH_MAX_RATIO - PINCH_MIN_RATIO), 0.0, 1.0)
                smooth_pinch = ema_pinch.update(raw_pinch_variable)
                
                pinch_color = (0, int(smooth_pinch * 255), int((1.0 - smooth_pinch) * 255))
                cv2.line(frame, (int(p4.x * w), int(p4.y * h)), (int(p8.x * w), int(p8.y * h)), pinch_color, 4)

                current_axes = get_orthogonal_axes(current_wrist, p5, p9, w, h)
                draw_axes(frame, current_wrist, current_axes, w, h)

                if baseline_wrist and baseline_axes and baseline_palm_size and arm_baseline_pose:
                    # Translation Math
                    t_x = (current_wrist.x - baseline_wrist.x) * w
                    t_y = (current_wrist.y - baseline_wrist.y) * h
                    depth_ratio = baseline_palm_size / (current_palm_size + 1e-6)
                    t_z = (depth_ratio - 1.0) * w 
                    
                    translation_vector_cam = np.array([t_x, t_y, t_z])
                    raw_dx = np.dot(translation_vector_cam, baseline_axes[0]) * SCALE_X
                    raw_dy = np.dot(translation_vector_cam, baseline_axes[1]) * SCALE_Y
                    raw_dz = np.dot(translation_vector_cam, baseline_axes[2]) * SCALE_Z
                    
                    smooth_dx = ema_dx.update(raw_dx)
                    smooth_dy = ema_dy.update(raw_dy)
                    smooth_dz = ema_dz.update(raw_dz)
                    
                    # Axis Angle Rotation Math
                    R_base = np.column_stack(baseline_axes)
                    R_curr = np.column_stack(current_axes)
                    R_rel = np.dot(R_curr, R_base.T)
                    
                    rvec, _ = cv2.Rodrigues(R_rel)
                    rvec = rvec.flatten()
                    
                    smooth_rx = ema_rx.update(rvec[0])
                    smooth_ry = ema_ry.update(rvec[1])
                    smooth_rz = ema_rz.update(rvec[2])
                    smooth_rvec = np.array([smooth_rx, smooth_ry, smooth_rz])
                    
                    angle = np.linalg.norm(smooth_rvec)
                    if angle > 1e-6:
                        axis = smooth_rvec / angle
                    else:
                        axis = np.array([0.0, 0.0, 1.0])
                        angle = 0.0
                    
                    draw_axes(frame, baseline_wrist, baseline_axes, w, h)
                    
                    # --- EXECUTE COMMANDS ONLY IF NOT PAUSED ---
                    if not movement_paused and (current_time_loop - last_command_time) > COMMAND_RATE_LIMIT:
                        
                        # GRIPPER CONTROL
                        if smooth_pinch < 0.25 and not is_gripping:
                            asyncio.create_task(gripper.grab())
                            is_gripping = True
                        elif smooth_pinch > 0.75 and is_gripping:
                            asyncio.create_task(gripper.open())
                            is_gripping = False

                        # ARM MOVEMENT
                        target_x = arm_baseline_pose.x + (smooth_dx * REAL_WORLD_SCALE)
                        target_y = arm_baseline_pose.y - (smooth_dy * REAL_WORLD_SCALE)
                        target_z = arm_baseline_pose.z + (smooth_dz * REAL_WORLD_SCALE)
                        
                        # Apply rotation logic based on the flag
                        if ENABLE_ROTATION:
                            target_o_x, target_o_y, target_o_z = axis[0], axis[1], axis[2]
                            target_theta = angle
                        else:
                            target_o_x, target_o_y, target_o_z = arm_baseline_pose.o_x, arm_baseline_pose.o_y, arm_baseline_pose.o_z
                            target_theta = arm_baseline_pose.theta

                        dist_moved = 0.0
                        if last_commanded_xyz:
                            dist_moved = math.sqrt(
                                (target_x - last_commanded_xyz[0])**2 +
                                (target_y - last_commanded_xyz[1])**2 +
                                (target_z - last_commanded_xyz[2])**2
                            )
                        else:
                            dist_moved = MOVE_THRESHOLD_MM + 1.0 

                        if dist_moved > MOVE_THRESHOLD_MM:
                            target_pose = Pose(
                                x=target_x,
                                y=target_y,
                                z=target_z,
                                o_x=target_o_x,
                                o_y=target_o_y,
                                o_z=target_o_z,
                                theta=target_theta
                            )
                            
                            asyncio.create_task(arm.move_to_position(
                                target_pose, 
                                extra={"velocity": MOVEMENT_VELOCITY}
                            ))
                            
                            last_commanded_xyz = (target_x, target_y, target_z)
                            last_command_time = current_time_loop

            cv2.imshow(window_name, frame)
            await asyncio.sleep(0.0001)

    cap.release()
    cv2.destroyAllWindows()
    print("Closing robot connection...")
    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())