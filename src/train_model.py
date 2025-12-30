import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import joblib
from utils import (load_config, save_model, evaluate_model, 
                  plot_confusion_matrix, plot_roc_curve, plot_feature_importance)
from data_preprocessing import DiabetesDataPreprocessor

class DiabetesModelTrainer:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.models = {}
        self.results = {}
        
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model"""
        print("\n" + "="*50)
        print("TRAINING LightGBM MODEL")
        print("="*50)
        
        params = self.config['model_params']['lightgbm']
        model = lgb.LGBMClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, 
                                   cv=self.config['training']['cv_folds'],
                                   scoring='roc_auc')
        print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Train final model
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        
        return model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("TRAINING XGBoost MODEL")
        print("="*50)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        return model
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for LightGBM"""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9],
            'num_leaves': [31, 50, 100],
        }
        
        base_model = lgb.LGBMClassifier(random_state=42)
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=5, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def evaluate_all_models(self, X_test, y_test, feature_names):
        """Evaluate all trained models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n{'-'*30}")
            print(f"Evaluating {name.upper()}")
            print(f"{'-'*30}")
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = evaluate_model(y_test, y_pred, y_pred_proba)
            self.results[name] = metrics
            
            # Plot confusion matrix for best model
            if name == 'lightgbm':
                plot_confusion_matrix(y_test, y_pred, 
                                     save_path=f"{self.config['paths']['results']}confusion_matrix.png")
                plot_roc_curve(y_test, y_pred_proba,
                              save_path=f"{self.config['paths']['results']}roc_curve.png")
                plot_feature_importance(model, feature_names,
                                       save_path=f"{self.config['paths']['results']}feature_importance.png")
    
    def select_best_model(self):
        """Select the best model based on ROC-AUC"""
        best_model_name = None
        best_score = 0
        
        for name, metrics in self.results.items():
            if 'roc_auc' in metrics and metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model_name = name
        
        if best_model_name:
            print(f"\n{'='*50}")
            print(f"BEST MODEL: {best_model_name.upper()} (ROC-AUC: {best_score:.4f})")
            print(f"{'='*50}")
            
            best_model = self.models[best_model_name]
            save_model(best_model, self.config['paths']['model'])
            
            return best_model
        return None
    
    def train_pipeline(self):
        """Complete training pipeline"""
        print("="*50)
        print("MODEL TRAINING PIPELINE")
        print("="*50)
        
        # Preprocess data
        preprocessor = DiabetesDataPreprocessor()
        X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline()
        
        # Train models
        self.train_lightgbm(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        
        # Optional: Hyperparameter tuning
        # best_lgb = self.hyperparameter_tuning(X_train, y_train)
        # self.models['lightgbm_tuned'] = best_lgb
        
        # Evaluate models
        self.evaluate_all_models(X_test, y_test, feature_names)
        
        # Select and save best model
        best_model = self.select_best_model()
        
        return best_model, X_test, y_test

if __name__ == "__main__":
    trainer = DiabetesModelTrainer()
    best_model, X_test, y_test = trainer.train_pipeline()