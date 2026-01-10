import random
import time
import pandas as pd
import numpy as np

# --- MODIFIED: FORCE A DANGER SPIKE ---
def generate_patient_data():
    """
    Simulates a live feed of patient vitals.
    Returns a dictionary of current readings.
    
    MODIFIED: We force a DANGER spike every 15 seconds (after the 15th read).
    """
    # Use st.session_state.data length (which tracks the number of reads)
    try:
        # Check if the monitoring loop has run for more than 15 seconds
        if st.session_state.data.shape[0] % 15 == 0 and st.session_state.data.shape[0] > 0:
            is_critical = True # FORCE IT
        else:
            is_critical = random.random() < 0.05 # Reduced random chance
    except NameError:
        # Fallback for when running outside Streamlit
        is_critical = random.random() < 0.15 

    if is_critical:
        # Generate 'Delirium/Psychosis' patterns (High HR, Low Oxygen, Agitation)
        hr = random.randint(110, 140)       # FORCED High Heart Rate
        spo2 = random.randint(85, 94)       # FORCED Low Oxygen
        sleep = random.randint(0, 1)        
        rass = random.randint(3, 4)         
    else:
        # Generate 'Normal' patterns
        hr = random.randint(65, 85)
        spo2 = random.randint(96, 100)
        sleep = random.randint(3, 5)
        rass = random.randint(-1, 0)

    return {
        'HR_Avg': hr,
        'SpO2_Min': spo2,
        'Sleep_Score': sleep,
        'RASS_Score': rass
    }

if __name__ == "__main__":
    # Test the generator (remains the same)
    print("Testing Sensor Simulation (Press Ctrl+C to stop)...")
    try:
        while True:
            data = generate_patient_data()
            print(f"💓 LIVE READ: {data}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")