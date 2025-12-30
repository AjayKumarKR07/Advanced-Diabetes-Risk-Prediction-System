import pandas as pd
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_model(model, filename):
    """Save trained model to file"""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """Load trained model from file"""
    return joblib.load(filename)

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Evaluate model performance"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper():<12}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), 
                  [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def create_interactive_dashboard(df):
    """Create interactive Plotly dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Glucose vs BMI', 'Age Distribution',
                       'Blood Pressure Distribution', 'Outcome Proportion'),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'histogram'}, {'type': 'pie'}]]
    )
    
    # Scatter plot
    fig.add_trace(
        go.Scatter(x=df['Glucose'], y=df['BMI'], mode='markers',
                  marker=dict(color=df['Outcome'], colorscale='RdYlGn_r',
                             showscale=True, colorbar=dict(title="Outcome"))),
        row=1, col=1
    )
    
    # Histograms
    fig.add_trace(go.Histogram(x=df['Age'], name='Age'), row=1, col=2)
    fig.add_trace(go.Histogram(x=df['BloodPressure'], name='BP'), row=2, col=1)
    
    # Pie chart
    outcome_counts = df['Outcome'].value_counts()
    fig.add_trace(
        go.Pie(labels=['Non-Diabetic', 'Diabetic'], 
              values=outcome_counts.values),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Diabetes Data Dashboard")
    return fig