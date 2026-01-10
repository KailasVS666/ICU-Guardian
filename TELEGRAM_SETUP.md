# 🚨 Telegram Alert Setup Guide

## Step 1: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/start` and then `/newbot`
3. Follow the prompts to name your bot (e.g., "ICU Guardian Bot")
4. You will receive a **Bot Token** (keep this secret!)

Example Bot Token: `123456789:ABCdefGHIjklmnoPQRstuvWXYZabcdefGH`

## Step 2: Get Your Chat ID

1. Send any message to your new bot in Telegram
2. Open this URL in your browser (replace TOKEN with your bot token):
   ```
   https://api.telegram.org/botTOKEN/getUpdates
   ```
3. Look for `"chat":{"id":123456789}` - that's your **Chat ID**

Example Chat ID: `123456789`

## Step 3: Configure Environment Variables (Windows PowerShell)

Run these commands in your terminal (they persist for the session):

```powershell
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
```

### For Persistent Configuration (Optional)

Add to your PowerShell profile:
```powershell
# Open notepad $PROFILE and add:
$env:TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
$env:TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
```

## Step 4: Test the Alert System

```powershell
python engine/alerts.py
```

You should see:
- ✅ Telegram Alert Sent messages in console
- 📱 Actual notifications in Telegram

## Step 5: Integrate into Vision Loop

In `pillars/self_harm.py`:

```python
from engine.alerts import process_smart_alert

# Inside the main loop:
if tube_alert:
    process_smart_alert("self_harm", True, threshold_seconds=3)
elif distress_alert:
    process_smart_alert("distress", True, threshold_seconds=3)
elif agitation_alert:
    process_smart_alert("agitation", True, threshold_seconds=5)
elif fall_alert:
    process_smart_alert("fall", True, threshold_seconds=1)
else:
    # No danger - reset timers
    process_smart_alert("self_harm", False)
    process_smart_alert("distress", False)
    process_smart_alert("agitation", False)
    process_smart_alert("fall", False)
```

## Alert Thresholds (Recommended)

| Pillar | Threshold | Reason |
|--------|-----------|--------|
| Self-Harm (Tube) | 1-2s | CRITICAL - immediate action needed |
| Distress (Fear/Sad) | 3-5s | Usually temporary emotions |
| Agitation | 5s | Normal patient movement |
| Fall (Bed Exit) | 1-2s | CRITICAL - patient safety |

## Security Notes

⚠️ **Never hardcode tokens in code!** Always use environment variables.

⚠️ **Bot tokens are secret** - if exposed, regenerate via @BotFather

⚠️ **Chat IDs are user identifiers** - keep private for HIPAA compliance

---

**Testing**: Use `engine/alerts.py` as standalone test without running full vision system.
