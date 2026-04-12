import requests
import socket
import sys

def check_connection():
    print("🔍 DIAGNOSTIC: Network Connectivity Check")
    print("-----------------------------------------")
    
    # 1. DNS Check
    try:
        print("\n1. Resolving api.telegram.org DNS...")
        ip = socket.gethostbyname("api.telegram.org")
        print(f"   ✅ DNS IP: {ip}")
    except Exception as e:
        print(f"   ❌ DNS Failed: {e}")
        
    # 2. General Internet Check (Google)
    try:
        print("\n2. Checking General Internet (google.com)...")
        r = requests.get("https://www.google.com", timeout=5)
        print(f"   ✅ Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")

    # 3. Telegram Check
    try:
        print("\n3. Checking Telegram API...")
        # Use verify=False to mimic our app's setting
        r = requests.get("https://api.telegram.org", timeout=5, verify=False)
        print(f"   ✅ Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")
        
    print("\n-----------------------------------------")
    print("INTERPRETATION:")
    print("If #2 works but #3 fails -> ISP/Firewall blocking Telegram. You NEED a VPN.")
    print("If #1 fails -> DNS issue. Try changing DNS to 8.8.8.8.")
    print("If all fail -> No internet access for Python.")

if __name__ == "__main__":
    check_connection()
