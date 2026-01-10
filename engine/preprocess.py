from sklearn.preprocessing import MinMaxScaler
import pandas as pd

DATA_PATH = 'data/sample_patient_data.csv'

def load_data(file_path=DATA_PATH):
    """
    Loads the patient data from a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the raw patient data.
    """
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Convert the Timestamp column to datetime objects
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        print(f"✅ Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None

def preprocess_data(df):
    """
    Scales numerical features and splits data into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): DataFrame loaded from load_data.
    
    Returns:
        tuple: (X_scaled, y) where X_scaled is the scaled feature array, 
               and y is the target array.
    """
    # Drop the Timestamp column as it's not a direct feature for ML
    df_features = df.drop(columns=['Timestamp', 'Psychosis_Label'])
    
    # Extract the target variable
    y = df['Psychosis_Label'].values
    
    # Initialize and apply the scaler (Normalizes data between 0 and 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    print(f"✅ Data preprocessed. X shape: {X_scaled.shape}, y shape: {y.shape}")
    return X_scaled, y

if __name__ == '__main__':
    # Test the data loading
    data = load_data()
    
    if data is not None:
        # Test the preprocessing and scaling
        X, y = preprocess_data(data)
        
        print("\nFirst 3 rows of Scaled Features (X):")
        print(X[:3])
        
        print("\nTarget Labels (y):")
        print(y)