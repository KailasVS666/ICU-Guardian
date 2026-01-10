import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
import mediapipe as mp

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from pillars.distress_ai import detect_distress  # TEMPORARILY DISABLED due to dependency conflicts
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

def calculate_mar(landmarks):
    # Landmarks for inner lip (MediaPipe Face Mesh)
    p1 = landmarks[78] 
    p2 = landmarks[308]
    dist = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    return dist

def run_vision_system():
    print("🔄 Initializing ICU Guardian Vision System...")
    
    # Load configuration
    config = get_vision_config()
    target_fps = config.fps_limit
    
    # Setup MediaPipe
    print("📦 Loading MediaPipe modules...")
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose 
    mp_face_mesh = mp.solutions.face_mesh 
    mp_drawing = mp.solutions.drawing_utils

    # Initialize models
    print("🧠 Loading AI models (this may take 30-60 seconds)...")
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    print("  ✓ Hand detection model loaded")
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    print("  ✓ Face detection model loaded")
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    print("  ✓ Pose estimation model loaded")
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    print("  ✓ Face mesh model loaded")

    camera_index = int(os.getenv("ICU_CAMERA_INDEX", "0"))
    print(f"📷 Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("❌ ERROR: Cannot access camera!")
        print("   Possible reasons:")
        print("   1. Camera is being used by another application")
        print("   2. Camera permissions denied")
        print("   3. No camera found")
        print("   4. Camera driver issue")
        return
    
    print("✅ Camera opened successfully!")
    
    # Initialize FPS limiter
    fps_limiter = FPSLimiter(target_fps=target_fps)
    
    # --- TRACKING VARIABLES ---
    prev_torso_pos = None 
    ACTIVITY_THRESHOLD = 0.06 
    AGITATION_THRESHOLD = 0.08  # Lowered threshold for agitation detection (was 0.15, too high)
    PERSISTENCE_FRAMES = 30  # Frames needed to confirm agitation
    distress_frame_count = 0 
    agitation_frame_count = 0  # Counter for agitation frames
    MAR_THRESHOLD = 0.05
    deepface_frame_counter = 0  # Counter for DeepFace (every 15 frames)
    detected_emotion = "None"
    emotion_confidence = 0.0
    is_distressed = False 
    prev_pose_landmarks = None  # Initialize pose tracking
    motion_history = []  # Initialize motion history
    
    print("📷 ICU Vision Guardian Active... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera read failed; exiting loop")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process all models
        face_results = face_detection.process(rgb_frame)
        hand_results = hands.process(rgb_frame)
        pose_results = pose.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame) 
        
        # --- TRACKING FLAGS ---
        tube_alert = False
        agitation_alert = False
        distress_alert = False
        fall_alert = False  
        face_box = None

        # --- DRAW SAFE ZONE (BED BOUNDARIES) ---
        safe_x_min, safe_x_max = int(w * 0.2), int(w * 0.8)
        safe_y_max = int(h * 0.85) 
        cv2.rectangle(frame, (safe_x_min, 0), (safe_x_max, safe_y_max), (0, 255, 255), 1)
        cv2.putText(frame, "SAFE BED ZONE", (safe_x_min + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # --- A. TUBE PROTECTION ---
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w); y_min = int(bboxC.ymin * h)
                width = int(bboxC.width * w); height = int(bboxC.height * h)
                pad = 50
                face_box = [max(0, x_min - pad), max(0, y_min - pad), 
                            min(w, x_min + width + pad), min(h, y_min + height + pad)]
                
                # Draw Box (Red)
                cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)
                cv2.putText(frame, "CRITICAL ZONE", (face_box[0], face_box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                break 

        # --- DRAW ARMS (POSE LANDMARKS) ---
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            # Draw arm connections with thick lines
            # Left arm: shoulder(11) -> elbow(13) -> wrist(15)
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
            left_elbow = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
            
            # Right arm: shoulder(12) -> elbow(14) -> wrist(16)
            right_shoulder = (int(landmarks[12].x * w), int(landmarks[12].y * h))
            right_elbow = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            right_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))
            
            # Draw left arm
            if landmarks[11].visibility > 0.5:
                cv2.line(frame, left_shoulder, left_elbow, (255, 0, 0), 4)  # Blue
                cv2.circle(frame, left_shoulder, 6, (255, 0, 0), -1)
                cv2.circle(frame, left_elbow, 6, (0, 165, 255), -1)  # Orange
            if landmarks[13].visibility > 0.5 and landmarks[15].visibility > 0.5:
                cv2.line(frame, left_elbow, left_wrist, (255, 0, 0), 4)
                cv2.circle(frame, left_wrist, 6, (0, 255, 0), -1)  # Green
            
            # Draw right arm
            if landmarks[12].visibility > 0.5:
                cv2.line(frame, right_shoulder, right_elbow, (255, 0, 0), 4)
                cv2.circle(frame, right_shoulder, 6, (255, 0, 0), -1)
                cv2.circle(frame, right_elbow, 6, (0, 165, 255), -1)
            if landmarks[14].visibility > 0.5 and landmarks[16].visibility > 0.5:
                cv2.line(frame, right_elbow, right_wrist, (255, 0, 0), 4)
                cv2.circle(frame, right_wrist, 6, (0, 255, 0), -1)
            
            cv2.putText(frame, "ARM DETECTION", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if hand_results.multi_hand_landmarks and face_box:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger = hand_landmarks.landmark[8] 
                ix, iy = int(index_finger.x * w), int(index_finger.y * h)
                if face_box[0] < ix < face_box[2] and face_box[1] < iy < face_box[3]:
                    tube_alert = True
        
        # --- B. AGITATION & FALL DETECTOR ---
        movement = 0.0  # Initialize movement for display
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            current_torso_pos = np.array([(shoulder_x + hip_x) / 2, (shoulder_y + hip_y) / 2])
            
            # Enhanced motion tracking for agitation
            if prev_pose_landmarks is not None:
                motion = 0.0
                # Track upper body motion (shoulders, elbows, wrists)
                motion_points = [11, 12, 13, 14, 15, 16]  # Both shoulders, elbows, wrists
                for idx in motion_points:
                    prev = prev_pose_landmarks[idx]
                    curr = landmarks[idx]
                    motion += abs(curr.x - prev.x) + abs(curr.y - prev.y)
                motion /= len(motion_points)  # Average motion
                
                motion_history.append(motion)
                if len(motion_history) > 10:
                    motion_history.pop(0)
                avg_motion = sum(motion_history) / len(motion_history) if motion_history else 0
                
                movement = avg_motion  # Update display value
                
                # Check for sustained agitation
                if avg_motion > AGITATION_THRESHOLD:
                    agitation_frame_count += 1
                    if agitation_frame_count >= PERSISTENCE_FRAMES:
                        agitation_alert = True
                else:
                    agitation_frame_count = max(0, agitation_frame_count - 1)
            
            prev_pose_landmarks = landmarks
            
            # Simple torso movement tracking
            if prev_torso_pos is not None:
                torso_movement = np.linalg.norm(current_torso_pos - prev_torso_pos)
                if torso_movement > ACTIVITY_THRESHOLD: agitation_alert = True
            prev_torso_pos = current_torso_pos

            hip_pixel_x = int(hip_x * w)
            hip_pixel_y = int(hip_y * h)
            cv2.circle(frame, (hip_pixel_x, hip_pixel_y), 8, (255, 0, 255), -1)

            if hip_pixel_x < safe_x_min or hip_pixel_x > safe_x_max or hip_pixel_y > safe_y_max:
                fall_alert = True

        # Display motion score and agitation status
        motion_color = (0, 0, 255) if movement > AGITATION_THRESHOLD else (255, 255, 0)
        cv2.putText(frame, f"Motion: {movement:.3f}", (w - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
        cv2.putText(frame, f"Threshold: {AGITATION_THRESHOLD:.3f}", (w - 200, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Agitation status indicator
        agitation_status = "🚨 AGITATION" if agitation_frame_count >= PERSISTENCE_FRAMES else "✅ Normal"
        status_color = (0, 0, 255) if agitation_frame_count >= PERSISTENCE_FRAMES else (0, 255, 0)
        cv2.putText(frame, agitation_status, (w - 200, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # --- C. DISTRESS DETECTOR ---
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            mar = calculate_mar(landmarks) * 100 
            cv2.putText(frame, f"MAR: {mar:.2f}", (w - 200, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if mar > MAR_THRESHOLD * 100:
                distress_frame_count += 1
                if distress_frame_count > 10: distress_alert = True
            else:
                distress_frame_count = max(0, distress_frame_count - 1)
        
        # --- C2. DEEPFACE EMOTION AI (Pillar 3) --- TEMPORARILY DISABLED
        # deepface_frame_counter += 1
        # if deepface_frame_counter % 15 == 0 and face_box:  # Run every 15 frames
        #     try:
        #         # Extract face ROI for DeepFace analysis
        #         x1, y1, x2, y2 = face_box
        #         face_roi_frame = frame[y1:y2, x1:x2].copy()
        #         
        #         if face_roi_frame.size > 0:
        #             result = detect_distress(frame, face_box)
        #             if result and result[0] != "Error":
        #                 detected_emotion, is_distressed = result
        #                 if is_distressed:
        #                     distress_alert = True
        #     except Exception as e:
        #         pass  # Silently handle any DeepFace errors 

        # --- ALERT LOGIC ---
        alert_text = ""
        alert_color = (0, 0, 0)
        any_danger = False # Flag to check if we should un-blur
        
        if tube_alert:
            alert_text = "CRITICAL: TUBE INTERFERENCE!"
            alert_color = (0, 0, 255) # Red
            any_danger = True
            safe_beep(2500, 100)
            process_smart_alert("self_harm", True, threshold_seconds=2)  # 2s threshold for CRITICAL
        elif fall_alert:
            alert_text = "[BED EXIT ATTEMPT]" # Removed Emoji
            alert_color = (0, 0, 139) # Dark Red
            any_danger = True
            safe_beep(1500, 200)
            process_smart_alert("fall", True, threshold_seconds=1)  # 1s threshold for CRITICAL
        elif agitation_alert:
            alert_text = "WARNING: HIGH AGITATION!"
            alert_color = (0, 165, 255) # Orange
            any_danger = True
            process_smart_alert("agitation", True, threshold_seconds=5)  # 5s threshold for movement
        elif distress_alert:
            alert_text = "NOTICE: SILENT DISTRESS!"
            alert_color = (0, 255, 255) # Yellow/Cyan
            any_danger = True
            process_smart_alert("distress", True, threshold_seconds=3)  # 3s threshold for emotions
        else:
            # No danger - reset all timers
            process_smart_alert("self_harm", False)
            process_smart_alert("fall", False)
            process_smart_alert("agitation", False)
            process_smart_alert("distress", False)

        # --- PRIVACY BLUR (NEW) ---
        # If there is NO danger, blur the face for privacy
        if not any_danger and face_box:
            # Extract the face region (ROI)
            roi = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
            if roi.size > 0:
                # Apply Gaussian Blur
                roi = cv2.GaussianBlur(roi, (51, 51), 30)
                # Put the blurred face back into the frame
                frame[face_box[1]:face_box[3], face_box[0]:face_box[2]] = roi
                
                # Label it
                cv2.putText(frame, "PRIVACY MODE ON", (face_box[0], face_box[1]-40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- HUD DISPLAY (MEDICAL MONITOR STYLE) ---
        # Display FPS
        current_fps = fps_limiter.get_current_fps()
        draw_hud_text(frame, f"FPS: {current_fps:.1f}", (10, h - 20), 
                     font_scale=0.5, color=(0, 255, 0))
        
        # Display detected emotion from DeepFace
        if detected_emotion != "None":
            emotion_text = f"Emotion: {detected_emotion.upper()}"
            emotion_color = (0, 255, 0) if not is_distressed else (0, 0, 255)
            cv2.putText(frame, emotion_text, (w - 300, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
            
            # Draw distress indicator
            if is_distressed:
                cv2.putText(frame, "⚠️ DISTRESS DETECTED", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        # Display main alert
        if alert_text:
            cv2.putText(frame, alert_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)
        
        # Update pose landmarks for next iteration
        if pose_results.pose_landmarks:
            prev_pose_landmarks = pose_results.pose_landmarks.landmark
            
        cv2.imshow('ICU Guardian - Computer Vision', frame)
        
        # Limit FPS
        fps_limiter.wait()
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import traceback
    from pathlib import Path
    
    try:
        run_vision_system()
    except Exception as e:
        # Log error to file for debugging
        error_log = Path(__file__).parent.parent / "logs" / "vision_error.log"
        error_log.parent.mkdir(exist_ok=True)
        
        with open(error_log, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Vision System Error: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
            f.write(f"\n{'='*50}\n\n")
        
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print(f"📝 Error logged to: {error_log}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)