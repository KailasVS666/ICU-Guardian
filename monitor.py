import time
import numpy as np
from datetime import datetime
from src.detect_model import load_trained_model, predict_psychosis
from src.simulation import generate_patient_data
from sklearn.preprocessing import MinMaxScaler

def main():
    print("==========================================")
    print("   🏥 ICU GUARDIAN - LIVE MONITORING      ")
    print("   (Press Ctrl + C to stop system)        ")
    print("==========================================")

    # 1. Load the AI Model
    print("[1] Loading AI Brain...")
    model = load_trained_model()
    
    if model is None:
        print("❌ Error: Model not found. Run 'main.py' first.")
        return

    print("[2] Connecting to Sensors...")
    time.sleep(1)
    print("✅ System Online. Monitoring started.\n")

    print(f"{'Time':<10} | {'HR':<4} | {'SpO2':<5} | {'RASS':<5} | {'Status':<15}")
    print("-" * 65)

    # Initialize scaler (Dummy fit to match training range for hackathon demo)
    scaler = MinMaxScaler()
    scaler.fit([[50, 80, 0, -5], [150, 100, 5, 4]]) 

    try:
        while True:
            # Get Live Data
            vitals = generate_patient_data()
            
            # Format for AI
            input_features = np.array([[
                vitals['HR_Avg'], vitals['SpO2_Min'], 
                vitals['Sleep_Score'], vitals['RASS_Score']
            ]])
            input_scaled = scaler.transform(input_features)

            # AI Prediction
            prediction, probability = predict_psychosis(model, input_scaled)
            
            # Visuals
            current_time = datetime.now().strftime("%H:%M:%S")
            
            if prediction == 1:
                status = "🔴 DANGER"
                color = "\033[91m" # Red
            elif probability > 0.4:
                status = "🟡 WARNING"
                color = "\033[93m" # Yellow
            else:
                status = "🟢 STABLE"
                color = "\033[92m" # Green
            
            reset = "\033[0m"

            print(f"{color}{current_time:<10} | {vitals['HR_Avg']:<4} | {vitals['SpO2_Min']:<5} | {vitals['RASS_Score']:<5} | {status:<15} ({probability:.0%}){reset}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring Stopped.")

if __name__ == "__main__":
    main()