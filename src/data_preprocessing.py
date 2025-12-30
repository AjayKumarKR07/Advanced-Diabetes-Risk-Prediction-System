import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from utils import load_config

class DiabetesDataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load dataset from CSV file"""
        df = pd.read_csv(self.config['paths']['data'])
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"Columns: {list(df.columns)}")
        print(f"\nMissing values per column:")
        print(df.isnull().sum())
        return df
    
    def handle_zeros(self, df):
        """
        Handle zeros in the dataset (which often represent missing values)
        Replace zeros with NaN for appropriate columns
        """
        # Columns where zero might be invalid/missing
        zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        # Replace zeros with NaN
        df[zero_cols] = df[zero_cols].replace(0, np.nan)
        
        return df
    
    def impute_missing_values(self, df):
        """Impute missing values with median"""
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Impute with median
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                print(f"Imputed {col} with median: {median_val:.2f}")
        
        return pd.concat([X, y], axis=1)
    
    def feature_engineering(self, df):
        """Create new features"""
        df_copy = df.copy()
        
        # Create BMI categories
        df_copy['BMI_Category'] = pd.cut(df_copy['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Create glucose categories
        df_copy['Glucose_Category'] = pd.cut(df_copy['Glucose'],
                                           bins=[0, 140, 200, 300],
                                           labels=['Normal', 'Prediabetes', 'Diabetes'])
        
        # Create age groups
        df_copy['Age_Group'] = pd.cut(df_copy['Age'],
                                     bins=[0, 30, 45, 60, 100],
                                     labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        # Interaction features
        df_copy['Glucose_BMI'] = df_copy['Glucose'] * df_copy['BMI']
        df_copy['Age_Glucose'] = df_copy['Age'] * df_copy['Glucose']
        
        # One-hot encode categorical features
        categorical_cols = ['BMI_Category', 'Glucose_Category', 'Age_Group']
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=True)
        
        return df_copy
    
    def split_data(self, df):
        """Split data into train and test sets"""
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        print(f"\nData split:")
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler for later use
        joblib.dump(self.scaler, self.config['paths']['scaler'])
        print(f"Scaler saved to {self.config['paths']['scaler']}")
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self):
        """Complete preprocessing pipeline"""
        print("="*50)
        print("DATA PREPROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Handle zeros/missing values
        df = self.handle_zeros(df)
        
        # Step 3: Impute missing values
        df = self.impute_missing_values(df)
        
        # Step 4: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\nPreprocessing completed successfully!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

if __name__ == "__main__":
    preprocessor = DiabetesDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline()