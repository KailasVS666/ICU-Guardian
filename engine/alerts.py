import time
import requests
import os
import json
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from enum import Enum
from typing import Optional

# Load environment variables from .env file
# Load environment variables from .env file
# Load environment variables from .env file
load_dotenv(override=True)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
print("DEBUG: Loaded env vars. Token present:", bool(os.getenv("TELEGRAM_BOT_TOKEN")))
print("DEBUG: User Chat ID present:", bool(os.getenv("TELEGRAM_CHAT_ID")))

# Path to shared status file
STATUS_FILE = Path(__file__).parent.parent / "alert_status.json"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


# Severity mapping for pillars
PILLAR_SEVERITY_MAP = {
    "distress": AlertSeverity.WARNING,
    "self_harm": AlertSeverity.CRITICAL,
    "agitation": AlertSeverity.HIGH,
    "fall": AlertSeverity.CRITICAL
}

# Import database
try:
    from engine.database import get_db
    db = get_db()
    DB_ENABLED = True
except ImportError:
    DB_ENABLED = False

# Track start times for persistent alerts
alert_timers = {
    "distress": None,
    "self_harm": None,
    "agitation": None,
    "fall": None
}

# Track last alert time to prevent spam
last_alert_time = {
    "distress": 0,
    "self_harm": 0,
    "agitation": 0,
    "fall": 0
}

# Track alert counts
alert_counts = {
    "distress": 0,
    "self_harm": 0,
    "agitation": 0,
    "fall": 0
}

# Track alert confidence scores
alert_confidence = {
    "distress": 0.0,
    "self_harm": 0.0,
    "agitation": 0.0,
    "fall": 0.0
}

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")  # Set via environment variable
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")      # Set via environment variable
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

def write_status_file():
    """Write current alert status to shared JSON file for dashboard."""
    try:
        status = {
            "last_updated": datetime.now().isoformat(),
            "pillars": {}
        }
        
        for pillar, start_time in alert_timers.items():
            severity = PILLAR_SEVERITY_MAP.get(pillar, AlertSeverity.WARNING).value
            if start_time is not None:
                elapsed = time.time() - start_time
                status["pillars"][pillar] = {
                    "active": True,
                    "elapsed_seconds": round(elapsed, 2),
                    "count": alert_counts[pillar],
                    "severity": severity,
                    "confidence": round(alert_confidence[pillar], 2),
                    "last_alert": datetime.fromtimestamp(last_alert_time[pillar]).isoformat() if last_alert_time[pillar] > 0 else None
                }
            else:
                status["pillars"][pillar] = {
                    "active": False,
                    "elapsed_seconds": 0,
                    "count": alert_counts[pillar],
                    "severity": severity,
                    "confidence": 0.0,
                    "last_alert": datetime.fromtimestamp(last_alert_time[pillar]).isoformat() if last_alert_time[pillar] > 0 else None
                }

        # Atomic write to avoid partial JSON reads by the backend
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = STATUS_FILE.with_suffix(STATUS_FILE.suffix + ".tmp")
        with open(tmp_path, 'w') as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, STATUS_FILE)
    except Exception as e:
        print(f"⚠️ Failed to write status file: {e}")

def process_smart_alert(pillar_name, is_active, threshold_seconds=3, confidence: Optional[float] = None):
    """
    Sends a Telegram alert only if the danger persists for threshold_seconds.
    Prevents alarm fatigue by throttling notifications.
    
    Args:
        pillar_name: "distress", "self_harm", "agitation", or "fall"
        is_active: Boolean indicating if the condition is currently detected
        threshold_seconds: Duration before triggering alert (default 3 seconds)
        confidence: Detection confidence score (0.0 to 1.0)
    """
    global alert_timers, last_alert_time, alert_confidence
    
    # Update confidence score
    if confidence is not None:
        alert_confidence[pillar_name] = max(0.0, min(1.0, confidence))
    elif is_active:
        alert_confidence[pillar_name] = 1.0
    else:
        alert_confidence[pillar_name] = 0.0
    
    if is_active:
        # Start timer if not already running
        if alert_timers[pillar_name] is None:
            alert_timers[pillar_name] = time.time()
        
        elapsed = time.time() - alert_timers[pillar_name]
        
        # Only send alert if threshold is reached AND enough time has passed since last alert
        if elapsed >= threshold_seconds:
            time_since_last = time.time() - last_alert_time[pillar_name]
            
            # Prevent spam: minimum 60 seconds between alerts for same pillar
            if time_since_last >= 60:
                alert_message = f"🚨 {pillar_name.upper()} ALERT: Persistent danger detected for {elapsed:.1f}s!"
                send_to_telegram(alert_message)
                last_alert_time[pillar_name] = time.time()
                alert_counts[pillar_name] += 1
                
                # Save to database
                if DB_ENABLED:
                    try:
                        # Determine severity based on elapsed time
                        if elapsed >= threshold_seconds * 3:
                            severity = "critical"
                        elif elapsed >= threshold_seconds * 2:
                            severity = "high"
                        else:
                            severity = "medium"
                        
                        # Get active session ID from backend (or create if needed)
                        from backend.main import app_state
                        session_id = app_state.ensure_session()
                        
                        db.save_alert(
                            session_id=session_id,
                            pillar=pillar_name,
                            severity=severity,
                            message=alert_message
                        )
                        print(f"💾 Alert saved to database (Session: {session_id})")
                    except Exception as e:
                        print(f"⚠️ Failed to save alert to database: {e}")
                
                # Reset timer to restart counting
                alert_timers[pillar_name] = time.time()
    else:
        # Condition cleared, reset timer
        alert_timers[pillar_name] = None
    
    # Write status to file after every update
    write_status_file()

def send_to_telegram(message):
    """
    Sends a message to Telegram with timestamp.
    
    Args:
        message: Alert message text
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"⚠️ Telegram not configured. Message: {message}")
        return False
    
    try:
        # Add timestamp to message
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": full_message,
            "parse_mode": "HTML"
        }
        
        # Check for proxy in env
        proxies = None
        if os.getenv("HTTPS_PROXY"):
            proxies = {"https": os.getenv("HTTPS_PROXY")}

        # Disable SSL verification to bypass potential proxy/firewall SSL inspection issues
        response = requests.post(
            TELEGRAM_API_URL, 
            json=payload, 
            timeout=5, 
            verify=False,
            proxies=proxies
        )
        
        if response.status_code == 200:
            print(f"✅ Telegram Alert Sent: {full_message}")
            return True
        else:
            print(f"⚠️ Telegram Error: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"🔕 Telegram blocked/unreachable. Alert logged locally only.")
        return False
    except Exception as e:
        print(f"⚠️ Failed to send Telegram message: {str(e)}")
        return False

def reset_all_timers():
    """Reset all alert timers (useful when patient is safe)."""
    global alert_timers, last_alert_time
    alert_timers = {
        "distress": None,
        "self_harm": None,
        "agitation": None,
        "fall": None
    }
    write_status_file()
    print("🔄 All alert timers reset.")

def get_alert_status():
    """Returns current status of all alerts."""
    status = {}
    for pillar, start_time in alert_timers.items():
        if start_time is not None:
            elapsed = time.time() - start_time
            status[pillar] = {"active": True, "elapsed_seconds": round(elapsed, 2)}
        else:
            status[pillar] = {"active": False, "elapsed_seconds": 0}
    return status

if __name__ == "__main__":
    # Test the alert system
    print("🧪 Testing ICU Guardian Alert System...")
    
    # Test 1: Single short burst (should not alert)
    print("\n📍 Test 1: Single burst (0.5s) - should NOT trigger alert")
    process_smart_alert("distress", True, threshold_seconds=3)
    time.sleep(0.5)
    process_smart_alert("distress", False, threshold_seconds=3)
    print(f"Status: {get_alert_status()}")
    
    # Test 2: Persistent activation (should alert after 3s)
    print("\n📍 Test 2: Persistent activation (4s) - should trigger alert")
    for i in range(4):
        process_smart_alert("self_harm", True, threshold_seconds=3)
        time.sleep(1)
        print(f"  → {i+1}s elapsed...")
    
    print(f"Status: {get_alert_status()}")
    reset_all_timers()
    print("✅ Alert system test complete!")
