import streamlit as st
import time
import pandas as pd
import numpy as np
import subprocess
import sys
from src.simulation import generate_patient_data
from src.detect_model import load_trained_model, predict_psychosis
from sklearn.preprocessing import MinMaxScaler

# Page Config
st.set_page_config(
    page_title="ICU Guardian",
    page_icon="🏥",
    layout="wide"
)

# Load Model Once
@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()

# Initialize Scaler (Same dummy fit as before for demo)
scaler = MinMaxScaler()
scaler.fit([[50, 80, 0, -5], [150, 100, 5, 4]])

# Title and Header
st.title("🏥 ICU Guardian: Advanced Psychosis Prevention")
st.markdown("---")

# Layout: 3 Columns
col1, col2, col3 = st.columns(3)

# Placeholders for Metrics
with col1:
    hr_metric = st.empty()
with col2:
    spo2_metric = st.empty()
with col3:
    status_metric = st.empty()

# Layout: Charts
st.markdown("### 📈 Live Vitals History")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    hr_chart = st.line_chart([])
with chart_col2:
    spo2_chart = st.line_chart([])

# Sidebar controls
st.sidebar.title("Controls")
if st.sidebar.button("📷 Launch Vision Guardian"):
    # Runs the vision script in a separate process
    subprocess.Popen([sys.executable, "src/vision_guardian.py"])
    st.sidebar.success("Vision System Launched!")

# Initialize Session State for Data History
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["HR", "SpO2", "Time"])

# Main Simulation Loop
start_button = st.button("▶️ Start Monitoring")

if start_button:
    # Run for a set number of iterations for the demo
    for i in range(100):
        # 1. Get Data
        vitals = generate_patient_data()
        
        # 2. Predict Risk
        input_features = np.array([[
            vitals['HR_Avg'], vitals['SpO2_Min'], 
            vitals['Sleep_Score'], vitals['RASS_Score']
        ]])
        input_scaled = scaler.transform(input_features)
        prediction, probability = predict_psychosis(model, input_scaled)

        # 3. Determine Status
        if prediction == 1:
            status_label = "CRITICAL RISK"
            status_color = "inverse" # Red-ish in Streamlit
        elif probability > 0.4:
            status_label = "WARNING"
            status_color = "off"
        else:
            status_label = "STABLE"
            status_color = "normal"

        # 4. Update Big Numbers (Metrics)
        hr_metric.metric("Heart Rate", f"{vitals['HR_Avg']} bpm", delta_color=status_color)
        spo2_metric.metric("SpO2", f"{vitals['SpO2_Min']} %")
        status_metric.error(f"Status: {status_label}") if prediction == 1 else status_metric.success(f"Status: {status_label}")

        # 5. Update Charts
        new_row = {"HR": vitals['HR_Avg'], "SpO2": vitals['SpO2_Min']}
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Keep only last 50 points to keep chart clean
        if len(st.session_state.data) > 50:
            st.session_state.data = st.session_state.data.iloc[1:]

        hr_chart.line_chart(st.session_state.data["HR"])
        spo2_chart.line_chart(st.session_state.data["SpO2"])

        time.sleep(1)