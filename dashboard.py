import streamlit as st
import time
import pandas as pd
import numpy as np
import subprocess
import sys
import random 
from src.simulation import generate_patient_data
from src.detect_model import load_trained_model, predict_psychosis
from sklearn.preprocessing import MinMaxScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ICU Guardian",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark Theme Setup */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E2130; border: 1px solid #2E3440;
        padding: 15px; border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.5);
    }
    /* Title */
    h1 { color: #00ADB5; font-family: 'Helvetica Neue', sans-serif; }
    /* Button */
    .stButton>button { background-color: #00ADB5; color: white; border: none; }
    .stButton>button:hover { background-color: #007F85; }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def get_model():
    return load_trained_model()
model = get_model()
scaler = MinMaxScaler()
# Initialize Scaler (Same dummy fit as before for demo)
scaler.fit([[50, 80, 0, -5], [150, 100, 5, 4]])

# --- SIMULATED AI DOCTOR (NEW) ---
def get_ai_analysis(vitals, risk_prob):
    """
    Simulates a Generative AI Doctor analyzing the patient and suggesting action.
    """
    hr = vitals['HR_Avg']
    spo2 = vitals['SpO2_Min']
    rass = vitals['RASS_Score']
    
    # 1. Critical Risk Check (from AI Model)
    if risk_prob > 0.4:
        return f"⚠️ **URGENT:** Patient showing signs of acute delirium (Risk: {risk_prob:.0%}). Vitals are unstable ({hr} bpm, RASS: {rass}). **Recommended Action:** Check for pain, assess sedation needs, and utilize the Vision Guardian to confirm agitation."
    
    # 2. Specific Vitals Check
    if spo2 < 95:
        return f"⚠️ **WARNING:** Hypoxia detected (SpO2: {spo2}%). Low oxygen can quickly lead to confusion. **Action:** Check oxygen mask/cannula immediately and reposition the patient."
    
    if hr > 100:
        return f"⚠️ **ALERT:** Tachycardia ({hr} bpm). Patient may be restless or in pain. Monitor movement and check temperature."
        
    # 3. Stable State
    return f"✅ **STATUS:** Patient is stable. Vitals are within normal limits (HR: {hr} BPM, RASS: {rass}). Continue standard observation."

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("ICU Guardian")
    st.markdown("---")
    if st.button("📷 Launch Vision System"):
        subprocess.Popen([sys.executable, "src/vision_guardian.py"])
        st.success("Vision System Active")
    st.markdown("---")
    st.caption("v1.1 | AI Nurse Assistant Active")

# --- MAIN LAYOUT ---
st.title("🏥 ICU Command Center")
col1, col2, col3, col4 = st.columns(4)
with col1: hr_metric = st.empty()
with col2: spo2_metric = st.empty()
with col3: sleep_metric = st.empty()
with col4: status_metric = st.empty()

# Layout: Charts vs AI Chat
col_charts, col_chat = st.columns([2, 1]) # Charts take 2/3rds, Chat takes 1/3rd

with col_charts:
    st.markdown("### 📊 Live Vitals Trends")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**Heart Rate (BPM)**")
        hr_chart = st.line_chart([])
    with chart_col2: 
        st.markdown("**Oxygen Saturation (%)**")
        spo2_chart = st.line_chart([])

with col_chat:
    st.markdown("### 🤖 Dr. AI Insights")
    ai_chat_box = st.empty() # Placeholder for AI messages

# Initialize History
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["HR", "SpO2", "Time"])

# --- MONITORING LOOP ---
monitoring = st.checkbox("▶️ Activate Monitoring", value=False)

if monitoring:
    while monitoring:
        # 1. Get Data & Predict
        vitals = generate_patient_data()
        input_scaled = scaler.transform(np.array([[
            vitals['HR_Avg'], vitals['SpO2_Min'], 
            vitals['Sleep_Score'], vitals['RASS_Score']
        ]]))
        prediction, probability = predict_psychosis(model, input_scaled)

        # 2. Update Metrics
        if prediction == 1:
            status_text, status_color = "CRITICAL RISK", "inverse"
            status_metric.error(f"Status: {status_text}")
        elif probability > 0.4:
            status_text, status_color = "WARNING", "off"
            status_metric.warning(f"Status: {status_text}")
        else:
            status_text, status_color = "STABLE", "normal"
            status_metric.success(f"Status: {status_text}")
            
        hr_metric.metric("Heart Rate", f"{vitals['HR_Avg']}", "BPM")
        spo2_metric.metric("SpO2", f"{vitals['SpO2_Min']}", "%")
        sleep_metric.metric("Sleep Score", f"{vitals['Sleep_Score']}/5")

        # 3. Update AI Doctor Chat
        ai_message = get_ai_analysis(vitals, probability)
        with ai_chat_box.container():
            st.info(ai_message, icon="🤖")

        # 4. Update Charts
        new_row = {"HR": vitals['HR_Avg'], "SpO2": vitals['SpO2_Min']}
        st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_row])], ignore_index=True)
        if len(st.session_state.data) > 50: st.session_state.data = st.session_state.data.iloc[1:]
        
        hr_chart.line_chart(st.session_state.data["HR"], height=200)
        spo2_chart.line_chart(st.session_state.data["SpO2"], height=200)

        time.sleep(1)