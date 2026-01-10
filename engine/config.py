"""
Configuration loader for ICU Guardian
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path(__file__).parent.parent / "config"
VISION_CONFIG_PATH = CONFIG_DIR / "vision.yaml"


class VisionConfig:
    """Vision system configuration"""
    
    def __init__(self, config_path: Path = VISION_CONFIG_PATH):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """Get config value by dot notation (e.g., 'camera.index')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    @property
    def camera_index(self) -> int:
        """Get camera index (env var overrides config)"""
        return int(os.getenv("ICU_CAMERA_INDEX", self.get("camera.index", 0)))
    
    @property
    def fps_limit(self) -> int:
        """Get target FPS limit"""
        return self.get("camera.fps_limit", 20)
    
    @property
    def agitation_motion_threshold(self) -> float:
        """Get agitation motion threshold"""
        return self.get("agitation.motion_threshold", 0.08)
    
    @property
    def agitation_persistence_frames(self) -> int:
        """Get agitation persistence frames"""
        return self.get("agitation.persistence_frames", 30)
    
    @property
    def safe_zone(self) -> Dict[str, float]:
        """Get safe zone boundaries"""
        return self.get("safe_zone", {
            "x_min_ratio": 0.2,
            "x_max_ratio": 0.8,
            "y_max_ratio": 0.85
        })
    
    @property
    def privacy_blur_enabled(self) -> bool:
        """Check if privacy blur is enabled"""
        return self.get("privacy.blur_face_when_safe", True)


# Global config instance
_vision_config = None


def get_vision_config() -> VisionConfig:
    """Get global vision config instance"""
    global _vision_config
    if _vision_config is None:
        _vision_config = VisionConfig()
    return _vision_config
