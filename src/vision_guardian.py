import cv2
import mediapipe as mp
import numpy as np
import winsound
import time

def calculate_mar(landmarks):
    # Landmarks for inner lip (MediaPipe Face Mesh)
    # 78: inner top lip center, 308: inner bottom lip center
    p1 = landmarks[78] 
    p2 = landmarks[308]
    
    # Vertical distance
    dist = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
    return dist

def run_vision_system():
    # 1. Setup MediaPipe for Hand, Face, and Pose
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
    ACTIVITY_THRESHOLD = 0.05
    distress_frame_count = 0 
    MAR_THRESHOLD = 0.04 
    
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
        face_box = None
        movement = 0  # <--- FIX: Initialized here to prevent UnboundLocalError

        # --- A. TUBE PROTECTION (Pillar 1) ---
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w); y_min = int(bboxC.ymin * h)
                width = int(bboxC.width * w); height = int(bboxC.height * h)
                pad = 50
                face_box = [max(0, x_min - pad), max(0, y_min - pad), 
                            min(w, x_min + width + pad), min(h, y_min + height + pad)]
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
        
        # --- B. AGITATION DETECTOR (Pillar 2) ---
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
            
        # Display Activity Score (Safe to draw now because movement is initialized to 0)
        cv2.putText(frame, f"Activity: {movement:.3f}", (w - 200, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- C. DISTRESS DETECTOR (Pillar 3) ---
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            mar = calculate_mar(landmarks) * 100 
            cv2.putText(frame, f"MAR: {mar:.2f}", (w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if mar > MAR_THRESHOLD * 100:
                distress_frame_count += 1
                if distress_frame_count > 10: 
                    distress_alert = True
            else:
                distress_frame_count = max(0, distress_frame_count - 1) 

        # --- D. FINAL ALERT DISPLAY ---
        alert_text = ""
        alert_color = (0, 0, 0)
        
        if tube_alert:
            alert_text = "🚨 CRITICAL: TUBE INTERFERENCE!"
            alert_color = (0, 0, 255) # Red
            winsound.Beep(2500, 100)
        elif agitation_alert:
            alert_text = "⚠️ WARNING: HIGH AGITATION!"
            alert_color = (0, 165, 255) # Orange
        elif distress_alert:
            alert_text = "😭 NOTICE: SILENT DISTRESS!"
            alert_color = (0, 255, 255) # Yellow/Cyan

        if alert_text:
             cv2.putText(frame, alert_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 3)
            
        cv2.imshow('ICU Guardian - Computer Vision', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_system()