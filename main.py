"""
Main entry point for Diabetes Prediction System
"""

import argparse
import sys
from src.data_preprocessing import DiabetesDataPreprocessor
from src.train_model import DiabetesModelTrainer
from src.predict import DiabetesPredictor

def main():
    parser = argparse.ArgumentParser(description="Diabetes Risk Prediction System")
    parser.add_argument('--mode', choices=['preprocess', 'train', 'predict', 'app'], 
                       default='app', help='Operation mode')
    parser.add_argument('--input', help='Input data for prediction (JSON file)')
    parser.add_argument('--output', help='Output file for predictions')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        print("Running data preprocessing...")
        preprocessor = DiabetesDataPreprocessor()
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline()
        print("Preprocessing completed!")
        
    elif args.mode == 'train':
        print("Training model...")
        trainer = DiabetesModelTrainer()
        trainer.train_pipeline()
        print("Training completed!")
        
    elif args.mode == 'predict':
        if not args.input:
            print("Error: --input required for prediction mode")
            sys.exit(1)
            
        import json
        with open(args.input, 'r') as f:
            patient_data = json.load(f)
        
        predictor = DiabetesPredictor()
        result = predictor.predict(patient_data)
        
        print("\nPrediction Results:")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    elif args.mode == 'app':
        print("Starting Streamlit web application...")
        import subprocess
        subprocess.run(["streamlit", "run", "src/app.py"])

if __name__ == "__main__":
    main()