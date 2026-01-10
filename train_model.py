"""
ML Model Training Script for ICU Guardian
Trains Random Forest model to predict psychosis/delirium risk from patient vitals
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import sys

def create_sample_data():
    """Generate synthetic training data if CSV doesn't exist"""
    print("📊 Generating synthetic training data...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Normal patients (label = 0)
    normal_data = {
        'HR_Avg': np.random.normal(75, 10, n_samples // 2),
        'SpO2_Min': np.random.normal(96, 2, n_samples // 2),
        'Sleep_Score': np.random.normal(3.5, 0.8, n_samples // 2),
        'RASS_Score': np.random.normal(0, 1, n_samples // 2),
        'Psychosis': 0
    }
    
    # At-risk patients (label = 1)
    risk_data = {
        'HR_Avg': np.random.normal(105, 15, n_samples // 2),
        'SpO2_Min': np.random.normal(90, 4, n_samples // 2),
        'Sleep_Score': np.random.normal(1.5, 0.9, n_samples // 2),
        'RASS_Score': np.random.normal(2, 1.5, n_samples // 2),
        'Psychosis': 1
    }
    
    # Combine and clip values to realistic ranges
    df_normal = pd.DataFrame(normal_data)
    df_risk = pd.DataFrame(risk_data)
    df = pd.concat([df_normal, df_risk], ignore_index=True)
    
    # Clip to realistic ranges
    df['HR_Avg'] = df['HR_Avg'].clip(50, 150)
    df['SpO2_Min'] = df['SpO2_Min'].clip(85, 100)
    df['Sleep_Score'] = df['Sleep_Score'].clip(0, 5)
    df['RASS_Score'] = df['RASS_Score'].clip(-5, 4)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def train_model():
    """Train and save the psychosis prediction model"""
    print("=" * 60)
    print("🧠 ICU GUARDIAN - ML MODEL TRAINING")
    print("=" * 60)
    
    # Create directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Load or create data
    data_file = data_dir / "sample_patient_data.csv"
    
    if data_file.exists():
        print(f"\n📁 Loading data from {data_file}...")
        df = pd.read_csv(data_file)
    else:
        print(f"\n⚠️  No data file found at {data_file}")
        df = create_sample_data()
        df.to_csv(data_file, index=False)
        print(f"✅ Generated and saved training data to {data_file}")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Target: {df.columns[-1]}")
    print(f"\n   Class distribution:")
    print(f"   - Normal (0): {(df['Psychosis'] == 0).sum()} samples ({(df['Psychosis'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"   - At-Risk (1): {(df['Psychosis'] == 1).sum()} samples ({(df['Psychosis'] == 1).sum() / len(df) * 100:.1f}%)")
    
    # Prepare features and labels
    X = df[['HR_Avg', 'SpO2_Min', 'Sleep_Score', 'RASS_Score']].values
    y = df['Psychosis'].values
    
    print(f"\n🔧 Preprocessing data...")
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train model
    print(f"\n🎯 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"   ✅ Model training complete!")
    
    # Evaluate model
    print(f"\n📈 Model Performance:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"   Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test set performance
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_accuracy = (y_pred == y_test).mean()
    print(f"   Test set accuracy: {test_accuracy:.3f}")
    
    # ROC-AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"   ROC-AUC Score: {auc_score:.3f}")
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'At-Risk']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  At-Risk")
    print(f"   Normal      {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"   At-Risk     {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Feature importance
    feature_names = ['HR_Avg', 'SpO2_Min', 'Sleep_Score', 'RASS_Score']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n🎯 Feature Importance:")
    for i, idx in enumerate(indices):
        print(f"   {i+1}. {feature_names[idx]:15s}: {importances[idx]:.3f}")
    
    # Save model
    model_path = models_dir / "psychosis_model.pkl"
    print(f"\n💾 Saving model to {model_path}...")
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"   ✅ Model saved successfully!")
    
    # Save scaler info
    scaler_path = models_dir / "scaler_params.pkl"
    print(f"\n💾 Saving scaler parameters to {scaler_path}...")
    
    scaler_info = {
        'data_min': scaler.data_min_,
        'data_max': scaler.data_max_,
        'feature_names': feature_names
    }
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_info, f)
    
    print(f"   ✅ Scaler parameters saved!")
    
    # Generate ROC curve plot
    print(f"\n📊 Generating ROC curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#5E6AD2', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='#8A8F98', linestyle='--', linewidth=1, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ICU Guardian - Psychosis Prediction ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = models_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150, facecolor='white')
    print(f"   ✅ ROC curve saved to {roc_path}")
    
    print(f"\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"=" * 60)
    print(f"\n📦 Generated files:")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")
    print(f"   - {roc_path}")
    print(f"\n🚀 You can now start the backend server!")
    print(f"   Run: python backend/main.py")
    print(f"\n")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
