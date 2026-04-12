import requests
import time
import json
import webbrowser
from pathlib import Path

# Configuration
TOKEN = "7977628604:AAGfkQL-FHproPVrvVq82QgIngjLDSlFEGg"
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

def setup_telegram():
    print(f"\n🤖 ICU Guardian Telegram Setup")
    print(f"--------------------------------")
    print(f"1. Open Telegram on your phone/desktop")
    print(f"2. Search for: @Icuguardian_bot")
    print(f"3. Click 'Start' or send the message '/start'")
    
    # Try automatic polling first
    print(f"\n⏳ Attempting to connect to Telegram API...")
    
    offset = 0
    consecutive_errors = 0
    
    while True:
        try:
            # Poll for updates with a timeout
            response = requests.get(
                f"{BASE_URL}/getUpdates", 
                params={"offset": offset},
                timeout=10
            ).json()
            
            # Reset error count on success
            consecutive_errors = 0
            
            if response.get("ok"):
                for result in response.get("result", []):
                    offset = result["update_id"] + 1
                    message = result.get("message")
                    if message:
                        handle_message(message)
                        return

            print(".", end="", flush=True)
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled.")
            break
        except Exception as e:
            consecutive_errors += 1
            # If we fail too many times, switch to manual mode
            if consecutive_errors >= 3:
                print(f"\n\n❌ Connection Error: {str(e)}")
                print("⚠️  It seems Python cannot reach Telegram servers (Firewall/ISP Block?).")
                manual_setup()
                break
            time.sleep(2)

def handle_message(message):
    chat_id = message["chat"]["id"]
    first_name = message["chat"].get("first_name", "User")
    
    print(f"\n\n✅ Message received from {first_name}!")
    print(f"🆔 YOUR CHAT ID: {chat_id}")
    
    # Send confirmation (best effort)
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": chat_id,
            "text": "✅ Connected to ICU Guardian!\nYou will now receive critical alerts here."
        }, timeout=5)
    except:
        pass
    
    # Update .env file
    update_env_file(str(chat_id))

def manual_setup():
    print(f"\n👇 MANUAL SETUP MODE")
    print(f"-------------------")
    print("1. Ensure you have sent a message to @Icuguardian_bot")
    print(f"2. Open this link in your web browser (you might need a VPN):")
    manual_url = f"{BASE_URL}/getUpdates"
    print(f"\n   {manual_url}\n")
    
    try:
        webbrowser.open(manual_url)
    except:
        pass

    print("3. Look for text like: \"chat\":{\"id\":123456789,...}")
    print("4. Copy the number (e.g. 123456789)")
    
    chat_id = input("\n📝 Enter your Chat ID here: ").strip()
    if chat_id:
        # Validate it's a number
        if chat_id.lstrip('-').isdigit():
            update_env_file(chat_id)
        else:
            print("❌ Invalid Chat ID type. Must be a number.")
    
def update_env_file(chat_id):
    """Updates the .env file with the new Chat ID"""
    env_path = Path(__file__).parent.parent / ".env"
    
    try:
        if env_path.exists():
            lines = env_path.read_text().splitlines()
            new_lines = []
            found = False
            
            for line in lines:
                if line.upper().startswith("TELEGRAM_CHAT_ID="):
                    new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
                    found = True
                else:
                    new_lines.append(line)
            
            if not found:
                new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
            
            env_path.write_text("\n".join(new_lines))
            print(f"\n💾 Updated configuration file: {env_path}")
            print("🎉 Setup Complete!")
            
    except Exception as e:
        print(f"❌ Failed to update .env: {e}")
        print(f"Please manually update .env with TELEGRAM_CHAT_ID={chat_id}")

if __name__ == "__main__":
    setup_telegram()
