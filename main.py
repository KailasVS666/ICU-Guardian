import pandas as pd
from src.preprocess import load_data, preprocess_data
from src.detect_model import load_trained_model, predict_psychosis, train_model, save_model

def main():
    print("==========================================")
    print("   🏥 ICU GUARDIAN - PSYCHOSIS DETECTION  ")
    print("==========================================")

    # 1. Load Data
    print("\n[1] Loading Patient Data...")
    df = load_data()
    
    if df is None:
        print("❌ System Aborted: No data found.")
        return

    # 2. Preprocess Data
    print("[2] Processing Vitals & Scores...")
    X, y = preprocess_data(df)

    # 3. Load AI Model (or train if missing)
    print("[3] Loading AI Model...")
    model = load_trained_model()
    
    if model is None:
        print("    ⚠️ Model not found. Training a new one...")
        model = train_model(X, y)
        save_model(model)

    # 4. Run Analysis on All Patients
    print("\n[4] 🔍 ANALYZING PATIENT RISKS:")
    print("-" * 50)
    print(f"{'Time':<25} | {'Risk Level':<15} | {'Probability':<10}")
    print("-" * 50)

    for i in range(len(X)):
        patient_features = X[i]
        timestamp = df.iloc[i]['Timestamp']
        
        prediction, probability = predict_psychosis(model, patient_features)
        
        # Determine Status Label
        if prediction == 1:
            status = "🔴 HIGH RISK"
        elif probability > 0.3:
            status = "🟡 WARNING"
        else:
            status = "🟢 NORMAL"

        print(f"{str(timestamp):<25} | {status:<15} | {probability:.2%}")

    print("-" * 50)
    print("✅ Analysis Complete.")

if __name__ == "__main__":
    main()