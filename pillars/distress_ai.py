from deepface import DeepFace
import cv2

def detect_distress(frame, face_box):
    """
    Analyzes the face to detect signs of pain or fear.
    """
    try:
        # 1. Crop the face (ROI)
        x1, y1, x2, y2 = face_box
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0: return None
        
        # 2. Run DeepFace Analysis
        # We use enforce_detection=False so it doesn't crash if the face is blurry
        analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        
        # 3. Detect dominant emotion
        dominant_emotion = analysis[0]['dominant_emotion']
        
        # Medical Logic: Fear or Sadness are proxies for Distress/Pain
        is_distressed = dominant_emotion in ['fear', 'sad']
        
        return dominant_emotion, is_distressed
    except Exception as e:
        return "Error", False
