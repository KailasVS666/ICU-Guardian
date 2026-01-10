"""
Agitation Detection Pillar
Detects rapid, erratic body movements indicating patient agitation
"""
import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
import time
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.alerts import process_smart_alert
from pillars.vision_utils import FPSLimiter, draw_hud_text
from engine.config import get_vision_config


def safe_beep(freq=1000, duration=150):
    """Cross-platform beep; no-op if unsupported."""
    try:
        import winsound  # type: ignore
        winsound.Beep(freq, duration)
    except Exception:
        pass

def run_agitation_detection():
    print("🚀 Initializing Agitation Detection...")
    
    # Load configuration
    config = get_vision_config()
    target_fps = config.fps_limit
    
    # Setup MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    print("🧠 Loading pose estimation model...")
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("  ✓ Pose model loaded")
    
    camera_index = int(os.getenv("ICU_CAMERA_INDEX", "0"))
    print(f"📷 Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    # Initialize FPS limiter
    fps_limiter = FPSLimiter(target_fps=target_fps)
    
    # Tracking variables
    prev_landmarks = None
    motion_history = []
    MOTION_THRESHOLD = 0.15
    PERSISTENCE_FRAMES = 30  # ~1 second at 30fps
    ALERT_COOLDOWN = 5
    last_alert_time = 0
    agitation_frame_count = 0
    
    print("📷 Agitation Guardian Active... Press 'q' to quit.")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera read failed; exiting loop")
                break
            
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = pose.process(rgb_frame)
            
            # Calculate motion
            motion = 0.0
            if results.pose_landmarks:
                curr_landmarks = results.pose_landmarks.landmark
                
                if prev_landmarks is not None:
                    # Calculate motion from key joints
                    key_indices = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
                    total_distance = 0
                    visible_count = 0
                    
                    for idx in key_indices:
                        if idx < len(curr_landmarks) and idx < len(prev_landmarks):
                            prev = prev_landmarks[idx]
                            curr = curr_landmarks[idx]
                            
                            if prev.visibility > 0.3 and curr.visibility > 0.3:
                                distance = np.sqrt(
                                    (curr.x - prev.x)**2 + 
                                    (curr.y - prev.y)**2
                                )
                                total_distance += distance
                                visible_count += 1
                    
                    if visible_count > 0:
                        motion = min(total_distance / visible_count, 1.0)
                
                motion_history.append(motion)
                if len(motion_history) > 10:
                    motion_history.pop(0)
                
                prev_landmarks = curr_landmarks
                
                # Check for agitation
                if len(motion_history) >= 3:
                    avg_motion = np.mean(motion_history[-5:])
                    
                    if avg_motion > MOTION_THRESHOLD:
                        agitation_frame_count += 1
                        
                        # Trigger alert if agitation persists
                        if agitation_frame_count > PERSISTENCE_FRAMES:
                            current_time = time.time()
                            if current_time - last_alert_time > ALERT_COOLDOWN:
                                print("🚨 AGITATION ALERT: Rapid body movements detected!")
                                try:
                                    process_smart_alert("agitation", "Patient showing rapid, erratic movements")
                                except:
                                    pass
                                safe_beep(1800, 120)
                                last_alert_time = current_time
                                agitation_frame_count = 0
                    else:
                        agitation_frame_count = max(0, agitation_frame_count - 2)
                
                # Draw pose
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display info
            status = "🚨 AGITATION" if agitation_frame_count > PERSISTENCE_FRAMES else "✅ Normal"
            color = (0, 0, 255) if agitation_frame_count > PERSISTENCE_FRAMES else (0, 255, 0)
            
            cv2.putText(frame, f"Motion: {np.mean(motion_history[-3:]) if motion_history else 0:.3f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display FPS
            current_fps = fps_limiter.get_current_fps()
            draw_hud_text(frame, f"FPS: {current_fps:.1f}", (10, h - 20), 
                         font_scale=0.5, color=(0, 255, 0))
            
            cv2.imshow('Agitation Detection', frame)
            
            # Limit FPS
            fps_limiter.wait()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("✅ Agitation detection stopped")
if __name__ == "__main__":
    run_agitation_detection()
