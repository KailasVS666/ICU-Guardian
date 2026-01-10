"""
Shared vision utilities for ICU Guardian pillars
Reusable components for camera management, motion tracking, and detection
"""
import cv2
import numpy as np
import time
import os
from typing import Optional, Tuple, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.config import get_vision_config


class SafeCamera:
    """
    Robust camera wrapper with auto-release and error handling
    """
    
    def __init__(self, camera_index: Optional[int] = None, max_retries: int = 3):
        self.config = get_vision_config()
        self.camera_index = camera_index if camera_index is not None else self.config.camera_index
        self.max_retries = max_retries
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_opened = False
    
    def open(self) -> bool:
        """Open camera with retry logic"""
        for attempt in range(self.max_retries):
            try:
                print(f"📷 Attempting to open camera {self.camera_index} (attempt {attempt + 1}/{self.max_retries})...")
                self.cap = cv2.VideoCapture(self.camera_index)
                
                if self.cap.isOpened():
                    # Test read
                    ret, _ = self.cap.read()
                    if ret:
                        self._is_opened = True
                        print(f"✅ Camera {self.camera_index} opened successfully")
                        return True
                    else:
                        self.cap.release()
                        print(f"⚠️ Camera opened but failed to read frame")
                
            except Exception as e:
                print(f"❌ Camera open attempt {attempt + 1} failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(1)
        
        print(f"❌ Failed to open camera {self.camera_index} after {self.max_retries} attempts")
        return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        if not self._is_opened or self.cap is None:
            return False, None
        
        try:
            return self.cap.read()
        except Exception as e:
            print(f"❌ Camera read error: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            try:
                self.cap.release()
                print("📷 Camera released")
            except Exception as e:
                print(f"⚠️ Error releasing camera: {e}")
            finally:
                self._is_opened = False
                self.cap = None
    
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        return self._is_opened and self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry"""
        if not self.open():
            raise RuntimeError(f"Failed to open camera {self.camera_index}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """Destructor"""
        self.release()


class MotionTracker:
    """
    Tracks body motion over time with smoothing and history
    """
    
    def __init__(self, history_size: int = 10, tracked_indices: Optional[List[int]] = None):
        self.config = get_vision_config()
        self.history_size = history_size
        self.tracked_indices = tracked_indices or self.config.get("agitation.tracking_points", [11, 12, 13, 14, 15, 16])
        self.motion_history: List[float] = []
        self.prev_landmarks = None
    
    def update(self, current_landmarks) -> float:
        """
        Calculate motion from current landmarks
        Returns: average motion score (0.0 to 1.0+)
        """
        if current_landmarks is None:
            return 0.0
        
        motion = 0.0
        
        if self.prev_landmarks is not None:
            total_distance = 0.0
            visible_count = 0
            
            for idx in self.tracked_indices:
                if idx < len(current_landmarks) and idx < len(self.prev_landmarks):
                    prev = self.prev_landmarks[idx]
                    curr = current_landmarks[idx]
                    
                    # Check visibility
                    if hasattr(prev, 'visibility') and hasattr(curr, 'visibility'):
                        if prev.visibility > 0.3 and curr.visibility > 0.3:
                            distance = np.sqrt(
                                (curr.x - prev.x)**2 + (curr.y - prev.y)**2
                            )
                            total_distance += distance
                            visible_count += 1
            
            if visible_count > 0:
                motion = min(total_distance / visible_count, 1.0)
        
        self.prev_landmarks = current_landmarks
        
        # Update history
        self.motion_history.append(motion)
        if len(self.motion_history) > self.history_size:
            self.motion_history.pop(0)
        
        return motion
    
    def get_average_motion(self, window: Optional[int] = None) -> float:
        """Get average motion over recent history"""
        if not self.motion_history:
            return 0.0
        
        window = window or len(self.motion_history)
        recent = self.motion_history[-window:]
        return np.mean(recent) if recent else 0.0
    
    def reset(self):
        """Reset tracking state"""
        self.motion_history.clear()
        self.prev_landmarks = None


class SafeZone:
    """
    Manages bed safe zone boundaries and fall detection
    """
    
    def __init__(self, frame_width: int, frame_height: int):
        self.config = get_vision_config()
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Load boundaries from config
        safe_zone = self.config.safe_zone
        self.x_min = int(frame_width * safe_zone["x_min_ratio"])
        self.x_max = int(frame_width * safe_zone["x_max_ratio"])
        self.y_max = int(frame_height * safe_zone["y_max_ratio"])
    
    def is_inside(self, x: int, y: int) -> bool:
        """Check if point is inside safe zone"""
        return self.x_min <= x <= self.x_max and y <= self.y_max
    
    def draw_on_frame(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255)):
        """Draw safe zone rectangle on frame"""
        cv2.rectangle(frame, (self.x_min, 0), (self.x_max, self.y_max), color, 2)
        cv2.putText(frame, "SAFE BED ZONE", (self.x_min + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class FPSLimiter:
    """
    Limits FPS to prevent CPU overload
    """
    
    def __init__(self, target_fps: Optional[int] = None):
        self.config = get_vision_config()
        self.target_fps = target_fps or self.config.fps_limit
        self.frame_duration = 1.0 / self.target_fps
        self.last_frame_time = time.time()
    
    def wait(self):
        """Wait to maintain target FPS"""
        elapsed = time.time() - self.last_frame_time
        if elapsed < self.frame_duration:
            time.sleep(self.frame_duration - elapsed)
        self.last_frame_time = time.time()
    
    def get_current_fps(self) -> float:
        """Calculate current FPS"""
        elapsed = time.time() - self.last_frame_time
        return 1.0 / elapsed if elapsed > 0 else 0.0


def apply_privacy_blur(frame: np.ndarray, face_box: Tuple[int, int, int, int], 
                       kernel_size: int = 51) -> np.ndarray:
    """
    Apply Gaussian blur to face region for privacy
    
    Args:
        frame: Input frame
        face_box: (x1, y1, x2, y2) bounding box
        kernel_size: Blur kernel size (must be odd)
    
    Returns:
        Frame with blurred face
    """
    x1, y1, x2, y2 = face_box
    
    # Validate coordinates
    if x1 >= x2 or y1 >= y2:
        return frame
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    if roi.size == 0:
        return frame
    
    # Apply blur
    try:
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 30)
        frame[y1:y2, x1:x2] = blurred_roi
    except Exception as e:
        print(f"⚠️ Privacy blur failed: {e}")
    
    return frame


def draw_hud_text(frame: np.ndarray, text: str, position: Tuple[int, int],
                  color: Tuple[int, int, int] = (255, 255, 255),
                  font_scale: float = 0.6, thickness: int = 2):
    """
    Draw HUD text with background for better visibility
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_height - 5), 
                 (x + text_width + 5, y + baseline + 5), 
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
