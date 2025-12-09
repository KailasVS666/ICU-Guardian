import cv2
import mediapipe as mp
import numpy as np
import winsound  # Windows-only sound library

def run_vision_system():
    # 1. Setup MediaPipe for Hand and Face detection
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Initialize models
    # High confidence thresholds to avoid false alarms
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

    # 2. Open Webcam
    cap = cv2.VideoCapture(0) # '0' is usually the default laptop camera

    print("📷 ICU Vision Guardian Active... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error")
            break

        # Flip image for natural mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Convert to RGB for MediaPipe (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- A. DETECT FACE (THE "CRITICAL ZONE") ---
        face_results = face_detection.process(rgb_frame)
        face_box = None

        if face_results.detections:
            for detection in face_results.detections:
                # Get bounding box of the face
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * w)
                y_min = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # Expand box slightly to cover neck/tubes area
                pad = 50 
                face_box = [max(0, x_min - pad), max(0, y_min - pad), 
                            min(w, x_min + width + pad), min(h, y_min + height + pad)]

                # Draw the "Red Zone"
                cv2.rectangle(frame, (face_box[0], face_box[1]), 
                              (face_box[2], face_box[3]), (0, 0, 255), 2)
                cv2.putText(frame, "CRITICAL ZONE (TUBES)", (face_box[0], face_box[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # We only need the first face for this demo
                break 

        # --- B. DETECT HANDS ---
        hand_results = hands.process(rgb_frame)
        alert_triggered = False

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if index finger tip is inside the Face Box
                if face_box:
                    # Landmark 8 is the Index Finger Tip
                    index_finger = hand_landmarks.landmark[8] 
                    ix, iy = int(index_finger.x * w), int(index_finger.y * h)

                    # Check collision: Is finger X/Y inside the Face Box?
                    if face_box[0] < ix < face_box[2] and face_box[1] < iy < face_box[3]:
                        alert_triggered = True

        # --- C. TRIGGER ALERT ---
        if alert_triggered:
            cv2.putText(frame, "⚠️ DANGER: HAND NEAR TUBES!", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Beep sound for alert
            winsound.Beep(2500, 100) # 2500Hz frequency, 100ms duration

        # Show the video feed
        cv2.imshow('ICU Guardian - Computer Vision', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vision_system()