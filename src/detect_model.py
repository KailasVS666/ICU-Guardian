import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Define paths
MODEL_PATH = 'models/psychosis_model.pkl'

def train_model(X, y):
    """
    Trains a Random Forest classifier.
    """
    print("🧠 Training AI Model...")
    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train on the provided data
    model.fit(X, y)

    print("✅ Model trained successfully.")
    return model

def save_model(model, path=MODEL_PATH):
    """
    Saves the trained model to a file.
    """
    joblib.dump(model, path)
    print(f"💾 Model saved to {path}")

def load_trained_model(path=MODEL_PATH):
    """
    Loads a trained model from disk.
    """
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"❌ No model found at {path}. Please train first.")
        return None

def predict_psychosis(model, patient_features):
    """
    Predicts probability of psychosis/delirium.
    """
    # Reshape input if it's a single sample (1D array -> 2D array)
    if patient_features.ndim == 1:
        patient_features = patient_features.reshape(1, -1)

    prediction = model.predict(patient_features)
    probability = model.predict_proba(patient_features)[0][1] # Probability of Class 1 (Psychosis)

    return prediction[0], probability

if __name__ == "__main__":
    # --- TEST RUN ---
    from preprocess import load_data, preprocess_data

    # 1. Get Data
    df = load_data()
    if df is not None:
        X, y = preprocess_data(df)

        # 2. Train Model
        model = train_model(X, y)

        # 3. Save Model
        save_model(model)

        # 4. Test Prediction (using the first patient's data)
        test_patient = X[0] 
        pred, prob = predict_psychosis(model, test_patient)
        print(f"\n🔎 Test Prediction for Patient 1:")
        print(f"   Prediction: {'Psychosis Detected' if pred == 1 else 'Normal'}")
        print(f"   Risk Probability: {prob:.2f}")