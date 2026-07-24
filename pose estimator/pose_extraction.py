import cv2
import time
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

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
        # alpha determines the smoothing (0.0 to 1.0)
        # 1.0 = No smoothing, 0.1 = Very heavy smoothing
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


def main():
    cap = cv2.VideoCapture(0)

    window_name = 'Pose Estimator'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    SCALE_X = 1.0 
    SCALE_Y = 1.0 
    SCALE_Z = 1.0 
    
    PINCH_MIN_RATIO = 0.15  
    PINCH_MAX_RATIO = 1.20  

    baseline_wrist = None
    baseline_axes = None
    baseline_palm_size = None
    
    is_logging = False
    log_file = None
    csv_writer = None
    video_writer = None

    # Instantiate filters for translation and pinch
    ema_dx = SimpleEMA(alpha=0.25)
    ema_dy = SimpleEMA(alpha=0.25)
    ema_dz = SimpleEMA(alpha=0.25)
    ema_pinch = SimpleEMA(alpha=0.25) 
    
    # Instantiate filters for rotation vectors
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
            
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key == ord('e'):
                if is_logging:
                    log_file.close()
                    video_writer.release()
                    is_logging = False
                    print("Logging ended.")
            elif key == ord('r'):
                current_result = latest_result 
                
                if current_result and current_result.hand_landmarks and len(current_result.hand_landmarks) > 0:
                    landmarks = current_result.hand_landmarks[0]
                    baseline_wrist = landmarks[0]
                    baseline_p5 = landmarks[5]
                    
                    baseline_axes = get_orthogonal_axes(baseline_wrist, baseline_p5, landmarks[9], w, h)
                    baseline_palm_size = np.sqrt(((baseline_p5.x - baseline_wrist.x)*w)**2 + ((baseline_p5.y - baseline_wrist.y)*h)**2)
                    
                    # Reset filters when zeroing
                    ema_dx.reset()
                    ema_dy.reset()
                    ema_dz.reset()
                    ema_pinch.reset()
                    ema_rx.reset()
                    ema_ry.reset()
                    ema_rz.reset()
                    
                    if not is_logging:
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        log_dir = os.path.join("logs", f"session_{timestamp_str}")
                        os.makedirs(log_dir, exist_ok=True)
                        
                        csv_path = os.path.join(log_dir, "pose_telemetry.csv")
                        log_file = open(csv_path, mode='w', newline='')
                        csv_writer = csv.writer(log_file)
                        
                        # Added Axis and Angle to header
                        csv_writer.writerow(['Timestamp', 'dX', 'dY', 'dZ', 'Pinch', 'Axis_X', 'Axis_Y', 'Axis_Z', 'theta'])
                        
                        video_path = os.path.join(log_dir, "screen_recording.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                        
                        is_logging = True
                        print(f"Baseline set. Logging to: {log_dir}")
    
            current_result = latest_result 
            
            # --- SAFETY CHECK: Hand tracking lost ---
            if not current_result or not current_result.hand_landmarks or len(current_result.hand_landmarks) == 0:
                # Clear the filter memory so it doesn't drag the arm when tracking returns
                ema_dx.reset()
                ema_dy.reset()
                ema_dz.reset()
                ema_pinch.reset()
                ema_rx.reset()
                ema_ry.reset()
                ema_rz.reset()
            else:
                # Hand is actively tracked
                hand_landmarks = current_result.hand_landmarks[0]
                
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]

                    start_x, start_y = int(start_point.x * w), int(start_point.y * h)
                    end_x, end_y = int(end_point.x * w), int(end_point.y * h)
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

                for landmark in hand_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                current_wrist = hand_landmarks[0]
                p5 = hand_landmarks[5]
                p9 = hand_landmarks[9]
                p4 = hand_landmarks[4]  
                p8 = hand_landmarks[8]  
                
                # Raw Pinch Calculation
                raw_pinch_dist = np.sqrt((p8.x - p4.x)**2 + (p8.y - p4.y)**2 + (p8.z - p4.z)**2)
                current_palm_size = np.sqrt(((p5.x - current_wrist.x)*w)**2 + ((p5.y - current_wrist.y)*h)**2)
                
                raw_pinch_ratio = raw_pinch_dist / ((current_palm_size / w) + 1e-6)
                raw_pinch_variable = np.clip((raw_pinch_ratio - PINCH_MIN_RATIO) / (PINCH_MAX_RATIO - PINCH_MIN_RATIO), 0.0, 1.0)
                
                # Filtered Pinch
                smooth_pinch = ema_pinch.update(raw_pinch_variable)
                
                pinch_color = (0, int(smooth_pinch * 255), int((1.0 - smooth_pinch) * 255))
                cv2.line(frame, 
                         (int(p4.x * w), int(p4.y * h)), 
                         (int(p8.x * w), int(p8.y * h)), 
                         pinch_color, 4)

                current_axes = get_orthogonal_axes(current_wrist, p5, p9, w, h)
                draw_axes(frame, current_wrist, current_axes, w, h)

                if baseline_wrist and baseline_axes and baseline_palm_size:
                    
                    # --- Translatiom Math ---
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
                    
                    # --- Axis Angle Rotation Math ---
                    # Create 3x3 rotation matrices from orthonormal axes 
                    R_base = np.column_stack(baseline_axes)
                    R_curr = np.column_stack(current_axes)
                    
                    # Relative rotation from baseline to current: R_rel = R_curr * R_base_inverse
                    R_rel = np.dot(R_curr, R_base.T)
                    
                    # Extract rotation vector (direction = axis, magnitude = angle)
                    rvec, _ = cv2.Rodrigues(R_rel)
                    rvec = rvec.flatten()
                    
                    # Apply EMA Filter to rotation vector
                    smooth_rx = ema_rx.update(rvec[0])
                    smooth_ry = ema_ry.update(rvec[1])
                    smooth_rz = ema_rz.update(rvec[2])
                    smooth_rvec = np.array([smooth_rx, smooth_ry, smooth_rz])
                    
                    # Extract final axis and angle representation
                    angle = np.linalg.norm(smooth_rvec)
                    if angle > 1e-6:
                        axis = smooth_rvec / angle
                    else:
                        axis = np.array([0.0, 0.0, 1.0])
                        angle = 0.0
                    
                    draw_axes(frame, baseline_wrist, baseline_axes, w, h)
                    
                    cv2.putText(frame, f"Local dX (Red): {smooth_dx:.2f}", (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Local dY (Grn): {smooth_dy:.2f}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Local dZ (Blu): {smooth_dz:.2f}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    if is_logging:
                        # Log data for Viam transmission (Includes axis-angle format now)
                        csv_writer.writerow([
                            time.time(), smooth_dx, smooth_dy, smooth_dz, smooth_pinch,
                            axis[0], axis[1], axis[2], angle
                        ])
                        cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)

            if is_logging and video_writer:
                video_writer.write(frame)
                
            cv2.imshow(window_name, frame)

    if is_logging:
        if log_file: log_file.close()
        if video_writer: video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()