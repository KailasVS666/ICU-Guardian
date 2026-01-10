"""
Simple camera test script
"""
import cv2
import sys

print("=" * 50)
print("CAMERA TEST")
print("=" * 50)

# Check OpenCV version
print(f"\n✓ OpenCV version: {cv2.__version__}")

# Try to open camera
print("\nAttempting to open camera (index 0)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Cannot open camera!")
    print("\nTroubleshooting steps:")
    print("1. Check if another application is using the camera")
    print("2. Check Windows Camera privacy settings:")
    print("   Settings > Privacy > Camera")
    print("3. Try unplugging and replugging external camera")
    print("4. Update camera drivers")
    print("5. Try a different camera index (1, 2, etc.)")
    sys.exit(1)

print("✅ Camera opened successfully!")

# Try to read a frame
ret, frame = cap.read()
if not ret:
    print("❌ ERROR: Can read from camera but cannot capture frames!")
    cap.release()
    sys.exit(1)

print(f"✅ Successfully captured frame: {frame.shape}")
print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
print(f"   Channels: {frame.shape[2]}")

# Release camera
cap.release()
print("\n✅ All tests passed! Camera is working correctly.")
print("\nPress any key to exit...")
input()
