import numpy as np
import pandas as pd
import joblib
from utils import load_config

class DiabetesPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        self.config = load_config()
        
        if model_path is None:
            model_path = self.config['paths']['model']
        if scaler_path is None:
            scaler_path = self.config['paths']['scaler']
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.thresholds = self.config['thresholds']
        
    def prepare_features(self, input_data):
        """Prepare input features for prediction"""
        # Create DataFrame with all expected features
        feature_template = {
            'Pregnancies': 0,
            'Glucose': 0,
            'BloodPressure': 0,
            'SkinThickness': 0,
            'Insulin': 0,
            'BMI': 0,
            'DiabetesPedigreeFunction': 0,
            'Age': 0
        }
        
        # Update with input data
        for key in input_data:
            if key in feature_template:
                feature_template[key] = input_data[key]
        
        # Create DataFrame
        df = pd.DataFrame([feature_template])
        
        # Feature engineering (same as in training)
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Glucose_Category'] = pd.cut(df['Glucose'],
                                       bins=[0, 140, 200, 300],
                                       labels=['Normal', 'Prediabetes', 'Diabetes'])
        df['Age_Group'] = pd.cut(df['Age'],
                                bins=[0, 30, 45, 60, 100],
                                labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        df['Glucose_BMI'] = df['Glucose'] * df['BMI']
        df['Age_Glucose'] = df['Age'] * df['Glucose']
        
        # One-hot encoding
        categorical_cols = ['BMI_Category', 'Glucose_Category', 'Age_Group']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Align columns with training data
        # In practice, you should save the training columns and use them here
        # For simplicity, we'll use the columns we expect
        
        return df
    
    def predict(self, input_data):
        """Make prediction for input data"""
        # Prepare features
        features_df = self.prepare_features(input_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        probability = self.model.predict_proba(features_scaled)[0, 1]
        
        # Classify based on threshold
        prediction = 1 if probability >= 0.5 else 0
        
        # Risk level
        if probability < self.thresholds['low_risk']:
            risk_level = "Low Risk"
            recommendation = "Maintain healthy lifestyle. Annual checkup recommended."
        elif probability < self.thresholds['high_risk']:
            risk_level = "Moderate Risk"
            recommendation = "Consult doctor. Monitor glucose levels regularly. Lifestyle changes needed."
        else:
            risk_level = "High Risk"
            recommendation = "Immediate medical consultation required. Further tests needed."
        
        result = {
            'probability': float(probability),
            'prediction': int(prediction),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'features_used': list(features_df.columns)
        }
        
        return result
    
    def batch_predict(self, df):
        """Make predictions for batch of data"""
        results = []
        for _, row in df.iterrows():
            result = self.predict(row.to_dict())
            results.append(result)
        
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Example usage
    predictor = DiabetesPredictor()
    
    # Test case 1: High risk patient
    test_patient_1 = {
        'Pregnancies': 2,
        'Glucose': 180,
        'BloodPressure': 85,
        'SkinThickness': 30,
        'Insulin': 150,
        'BMI': 35.5,
        'DiabetesPedigreeFunction': 0.8,
        'Age': 55
    }
    
    # Test case 2: Low risk patient
    test_patient_2 = {
        'Pregnancies': 0,
        'Glucose': 90,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 80,
        'BMI': 22.5,
        'DiabetesPedigreeFunction': 0.3,
        'Age': 25
    }
    
    print("="*50)
    print("DIABETES RISK PREDICTION")
    print("="*50)
    
    for i, patient in enumerate([test_patient_1, test_patient_2], 1):
        result = predictor.predict(patient)
        print(f"\nPatient {i} Prediction:")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Prediction: {'Diabetic' if result['prediction'] else 'Non-Diabetic'}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        print("-"*50)