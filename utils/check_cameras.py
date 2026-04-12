import cv2
import time

def list_cameras():
    print("🔍 Scanning for available cameras (Indices 0-5)...")
    available_cameras = []
    
    for index in range(5):
        try:
            # Try to open the camera (DirectShow for Windows)
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    h, w, c = frame.shape
                    print(f"✅ Camera found at Index {index}: Resolution {w}x{h}")
                    available_cameras.append(index)
                    
                    # Optional: Save a snapshot to verify visually which one it is
                    # timestamp = int(time.time())
                    # cv2.imwrite(f"camera_test_idx_{index}_{timestamp}.jpg", frame)
                else:
                    print(f"⚠️  Camera at Index {index} opened but returned no frame.")
                cap.release()
            else:
                print(f"❌ No camera at Index {index}")
                
        except Exception as e:
            print(f"❌ Error checking Index {index}: {e}")

    print("\n📋 Summary of Available Cameras:")
    if available_cameras:
        for idx in available_cameras:
            print(f" - Index {idx}")
        print(f"\n💡 Try changing 'camera_index' in self_harm.py to one of these.")
    else:
        print("❌ No working cameras found.")

if __name__ == "__main__":
    list_cameras()
