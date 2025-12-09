import random
import time

def generate_patient_data():
    """
    Simulates a live feed of patient vitals.
    Returns a dictionary of current readings.
    """
    # Simulate mostly normal data, with occasional 'risk' events (15% chance)
    is_critical = random.random() < 0.15

    if is_critical:
        # Generate 'Delirium/Psychosis' patterns (High HR, Low Oxygen, Agitation)
        hr = random.randint(100, 140)       
        spo2 = random.randint(85, 94)       
        sleep = random.randint(0, 2)        
        rass = random.randint(2, 4)         
    else:
        # Generate 'Normal' patterns
        hr = random.randint(60, 90)
        spo2 = random.randint(95, 100)
        sleep = random.randint(3, 5)
        rass = random.randint(-1, 1)

    return {
        'HR_Avg': hr,
        'SpO2_Min': spo2,
        'Sleep_Score': sleep,
        'RASS_Score': rass
    }

if __name__ == "__main__":
    # Test the generator loop
    print("Testing Sensor Simulation (Press Ctrl+C to stop)...")
    try:
        while True:
            data = generate_patient_data()
            print(f"💓 LIVE READ: {data}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")