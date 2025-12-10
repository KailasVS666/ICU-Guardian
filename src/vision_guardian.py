import cv2
import mediapipe as mp
import numpy as np
import winsound
import time

def calculate_mar(landmarks):
    # Landmarks for inner lip (MediaPipe Face Mesh)
    p1 = landmarks[78] 
    p2 = landmarks[308]
    dist = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    return dist

def run_vision_system():
    # Setup MediaPipe
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose 
    mp_face_mesh = mp.solutions.face_mesh 
    mp_drawing = mp.solutions.drawing_utils

    # Initialize models
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    cap = cv2.VideoCapture(0) 
    
    # --- TRACKING VARIABLES ---
    prev_torso_pos = None 
    ACTIVITY_THRESHOLD = 0.06 
    distress_frame_count = 0 
    MAR_THRESHOLD = 0.05 
    
    print("📷 ICU Vision Guardian Active... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

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
        movement = 0

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

        if hand_results.multi_hand_landmarks and face_box:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger = hand_landmarks.landmark[8] 
                ix, iy = int(index_finger.x * w), int(index_finger.y * h)
                if face_box[0] < ix < face_box[2] and face_box[1] < iy < face_box[3]:
                    tube_alert = True
        
        # --- B. AGITATION & FALL DETECTOR ---
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            hip_x = (landmarks[23].x + landmarks[24].x) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            current_torso_pos = np.array([(shoulder_x + hip_x) / 2, (shoulder_y + hip_y) / 2])
            
            if prev_torso_pos is not None:
                movement = np.linalg.norm(current_torso_pos - prev_torso_pos)
                if movement > ACTIVITY_THRESHOLD: agitation_alert = True
            prev_torso_pos = current_torso_pos

            hip_pixel_x = int(hip_x * w)
            hip_pixel_y = int(hip_y * h)
            cv2.circle(frame, (hip_pixel_x, hip_pixel_y), 8, (255, 0, 255), -1)

            if hip_pixel_x < safe_x_min or hip_pixel_x > safe_x_max or hip_pixel_y > safe_y_max:
                fall_alert = True

        cv2.putText(frame, f"Activity: {movement:.3f}", (w - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- C. DISTRESS DETECTOR ---
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            mar = calculate_mar(landmarks) * 100 
            cv2.putText(frame, f"MAR: {mar:.2f}", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if mar > MAR_THRESHOLD * 100:
                distress_frame_count += 1
                if distress_frame_count > 10: distress_alert = True
            else:
                distress_frame_count = max(0, distress_frame_count - 1) 

        # --- ALERT LOGIC ---
        alert_text = ""
        alert_color = (0, 0, 0)
        any_danger = False # Flag to check if we should un-blur
        
        if tube_alert:
            alert_text = "CRITICAL: TUBE INTERFERENCE!"
            alert_color = (0, 0, 255) # Red
            any_danger = True
            winsound.Beep(2500, 100)
        elif fall_alert:
            alert_text = "[BED EXIT ATTEMPT]" # Removed Emoji
            alert_color = (0, 0, 139) # Dark Red
            any_danger = True
            winsound.Beep(1500, 200)
        elif agitation_alert:
            alert_text = "WARNING: HIGH AGITATION!"
            alert_color = (0, 165, 255) # Orange
            any_danger = True
        elif distress_alert:
            alert_text = "NOTICE: SILENT DISTRESS!"
            alert_color = (0, 255, 255) # Yellow/Cyan
            any_danger = True

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

        if alert_text:
             cv2.putText(frame, alert_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)
            
        cv2.imshow('ICU Guardian - Computer Vision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_system()