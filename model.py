import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class AnomalyDetector:
    def __init__(self):
        self.categorical_features = ['Merchant', 'TransactionType', 'Location', 'AccountID']
        self.numerical_features = ['Amount', 'Unix_Timestamp']
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.scaler, self.numerical_features),
                ('cat', self.encoder, self.categorical_features)
            ]
        )
        self.model = IsolationForest(random_state=42)
    
    def load_and_clean_data(self, filepath):
        """Loads and cleans the dataset, ensuring correct datetime parsing."""
        df = pd.read_csv(filepath, low_memory=False)
        
        # Convert 'Timestamp' to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M', dayfirst=True, errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        
        # Convert 'Amount' to numeric
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df.dropna(subset=['Amount'], inplace=True)
        
        # Convert 'Timestamp' to Unix timestamp (float64)
        df['Unix_Timestamp'] = df['Timestamp'].map(lambda x: x.timestamp())
        
        return df
    
    def train(self, filepath):
        df = self.load_and_clean_data(filepath)
        if df is None or df.empty:
            print("⚠️ No valid data to train on.")
            return
        
        df_processed = self.preprocessor.fit_transform(df)

        # Train Isolation Forest with more estimators and defined contamination
        self.model = IsolationForest(n_estimators=10000, contamination=0.001, random_state=42)
        self.model.fit(df_processed)
        
        df['Is_Anomaly'] = self.model.predict(df_processed)

        # Ensure -1 is for anomalies
        df['Is_Anomaly'] = df['Is_Anomaly'].map({1: 0, -1: 1})

        print(f"🔍 Total anomalies detected: {df['Is_Anomaly'].sum()}")

        # Save model and preprocessors
        os.makedirs('model', exist_ok=True)
        dump(self.model, 'model/model.joblib')
        dump(self.encoder, 'model/encoder.joblib')
        dump(self.scaler, 'model/scaler.joblib')
        dump(self.preprocessor, 'model/preprocessor.joblib')

        # Save only anomalies
        anomalies = df[df['Is_Anomaly'] == 1]
        os.makedirs('data', exist_ok=True)
        anomalies.to_csv('data/anomalies.csv', index=False)

        print(anomalies.head())  # Debugging output

        return anomalies

    
    def detect(self, df):
        if not os.path.exists('model/model.joblib'):
            print("Model not found. Training a new model...")
            return self.train('data/data.csv')
        
        # Load models
        self.model = load('model/model.joblib')
        self.preprocessor = load('model/preprocessor.joblib')
        
        # Only load the dataset if df is a file path
        if isinstance(df, str):  
            df = self.load_and_clean_data(df)
        
        if not isinstance(df, pd.DataFrame):  
            raise ValueError("Invalid input: 'df' must be a DataFrame or a valid file path.")
        
        df_processed = self.preprocessor.transform(df)
        df['Is_Anomaly'] = self.model.predict(df_processed)
        
        # Return only anomalies (Is_Anomaly == -1)
        anomalies = df[df['Is_Anomaly'] == -1]
        
        # Save anomalies to CSV
        os.makedirs('data', exist_ok=True)
        anomalies.to_csv('data/anomalies.csv', index=False)
        
        return anomalies

if __name__ == "__main__":
    detector = AnomalyDetector()
    anomalies = detector.train('data/data.csv')
    if anomalies is not None:
        print(anomalies.head())
