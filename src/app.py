import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# ============================================
# ENHANCED PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Advanced Diabetes Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/AjayKumarKR07/diabetes-prediction',
        'Report a bug': "https://github.com/AjayKumarKR07/diabetes-prediction/issues",
        'About': "## Diabetes Risk Prediction System\n### ML Project with Advanced Features"
    }
)

# ============================================
# CUSTOM CSS FOR PROFESSIONAL UI
# ============================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
    }
    
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--primary);
        border-left: 4px solid var(--primary);
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
        margin: 1rem 0;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    /* Risk level indicators */
    .risk-low {
        color: var(--success);
        font-weight: bold;
        font-size: 1.3rem;
        padding: 0.5rem 1rem;
        background: rgba(16, 185, 129, 0.1);
        border-radius: 10px;
        display: inline-block;
    }
    
    .risk-medium {
        color: var(--warning);
        font-weight: bold;
        font-size: 1.3rem;
        padding: 0.5rem 1rem;
        background: rgba(245, 158, 11, 0.1);
        border-radius: 10px;
        display: inline-block;
    }
    
    .risk-high {
        color: var(--danger);
        font-weight: bold;
        font-size: 1.3rem;
        padding: 0.5rem 1rem;
        background: rgba(239, 68, 68, 0.1);
        border-radius: 10px;
        display: inline-block;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary);
    }
    
    /* Disease recommendation cards */
    .disease-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.8rem 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .disease-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .disease-critical {
        border-left-color: #EF4444;
        background: rgba(239, 68, 68, 0.05);
    }
    
    .disease-warning {
        border-left-color: #F59E0B;
        background: rgba(245, 158, 11, 0.05);
    }
    
    .disease-info {
        border-left-color: #2E86AB;
        background: rgba(46, 134, 171, 0.05);
    }
    
    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Custom slider styling */
    .stSlider > div > div > div {
        background: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DISEASE RECOMMENDATION DATABASE
# ============================================
def get_disease_recommendations(parameters):
    """
    Generate specific disease recommendations based on patient parameters
    Returns a list of dictionaries with disease info and recommendations
    """
    glucose = parameters.get('Glucose', 0)
    bmi = parameters.get('BMI', 0)
    blood_pressure = parameters.get('BloodPressure', 0)
    age = parameters.get('Age', 0)
    insulin = parameters.get('Insulin', 0)
    
    recommendations = []
    
    # 1. DIABETES-RELATED CONDITIONS
    if glucose >= 126:
        recommendations.append({
            'disease': 'Type 2 Diabetes',
            'risk': 'High' if glucose > 200 else 'Moderate',
            'symptoms': ['Frequent urination', 'Increased thirst', 'Fatigue', 'Blurred vision'],
            'tests': ['HbA1c test', 'Fasting plasma glucose', 'Oral glucose tolerance test'],
            'actions': [
                'Consult endocrinologist immediately',
                'Start blood glucose monitoring',
                'Begin dietary modifications (low glycemic index foods)',
                'Consider metformin therapy if prescribed'
            ],
            'severity': 'critical',
            'icon': '🩸'
        })
    
    elif glucose >= 100:
        recommendations.append({
            'disease': 'Prediabetes',
            'risk': 'Moderate',
            'symptoms': ['Often asymptomatic', 'Mild fatigue', 'Occasional thirst'],
            'tests': ['HbA1c (5.7-6.4%)', 'Fasting glucose test'],
            'actions': [
                'Lifestyle intervention program',
                'Weight reduction 5-7% if overweight',
                '150 minutes moderate exercise weekly',
                'Reduce refined carbohydrates'
            ],
            'severity': 'warning',
            'icon': '⚠️'
        })
    
    # 2. CARDIOVASCULAR CONDITIONS
    if blood_pressure >= 140:
        recommendations.append({
            'disease': 'Hypertension (Stage 2)',
            'risk': 'High',
            'symptoms': ['Headaches', 'Shortness of breath', 'Nosebleeds'],
            'tests': ['24-hour ambulatory BP monitoring', 'Echocardiogram', 'Renal function tests'],
            'actions': [
                'Emergency medical evaluation needed',
                'Start antihypertensive medication',
                'Reduce sodium intake (<1500mg/day)',
                'Regular BP monitoring 2x daily'
            ],
            'severity': 'critical',
            'icon': '❤️'
        })
    elif blood_pressure >= 130:
        recommendations.append({
            'disease': 'Hypertension (Stage 1)',
            'risk': 'Moderate',
            'symptoms': ['Usually asymptomatic', 'Mild headaches'],
            'tests': ['Multiple BP readings', 'Basic metabolic panel'],
            'actions': [
                'Consult cardiologist within 1 week',
                'DASH diet implementation',
                'Aerobic exercise 30 mins daily',
                'Limit alcohol to 1 drink/day'
            ],
            'severity': 'warning',
            'icon': '💓'
        })
    
    # 3. METABOLIC SYNDROME CONDITIONS
    if bmi >= 30:
        recommendations.append({
            'disease': 'Obesity-related Complications',
            'risk': 'High',
            'symptoms': ['Joint pain', 'Sleep apnea', 'Fatigue', 'Shortness of breath'],
            'tests': ['Lipid profile', 'Liver function tests', 'Sleep study if indicated'],
            'actions': [
                'Comprehensive weight management program',
                'Bariatric surgery evaluation if BMI > 35 with comorbidities',
                'Nutritionist consultation',
                'Screen for depression/anxiety'
            ],
            'severity': 'critical',
            'icon': '⚖️'
        })
    elif bmi >= 25:
        recommendations.append({
            'disease': 'Overweight-related Risks',
            'risk': 'Moderate',
            'symptoms': ['Reduced mobility', 'Increased fatigue'],
            'tests': ['Waist circumference measurement', 'Body composition analysis'],
            'actions': [
                'Structured weight loss program',
                'Calorie deficit of 500 kcal/day',
                'Resistance training 2x/week',
                'Behavioral therapy for eating habits'
            ],
            'severity': 'warning',
            'icon': '📊'
        })
    
    # 4. AGE-RELATED CONDITIONS
    if age >= 60:
        recommendations.append({
            'disease': 'Age-related Metabolic Decline',
            'risk': 'Moderate',
            'symptoms': ['Reduced muscle mass', 'Decreased mobility', 'Cognitive changes'],
            'tests': ['Comprehensive geriatric assessment', 'Bone density scan', 'Cognitive screening'],
            'actions': [
                'Geriatric medicine consultation',
                'Strength training to prevent sarcopenia',
                'Vitamin D and calcium supplementation',
                'Regular cognitive stimulation activities'
            ],
            'severity': 'info',
            'icon': '👴'
        })
    
    # 5. INSULIN RESISTANCE CONDITIONS
    if insulin >= 150:
        recommendations.append({
            'disease': 'Insulin Resistance Syndrome',
            'risk': 'High',
            'symptoms': ['Acanthosis nigricans', 'Skin tags', 'Fatigue after meals'],
            'tests': ['HOMA-IR calculation', 'Fasting insulin level', 'Lipid profile'],
            'actions': [
                'Endocrinology referral',
                'Low-carbohydrate diet (<130g/day)',
                'High-intensity interval training',
                'Consider metformin or GLP-1 agonists'
            ],
            'severity': 'critical',
            'icon': '🔄'
        })
    
    # 6. GENERAL HEALTH MAINTENANCE (always included)
    recommendations.append({
        'disease': 'Preventive Health Maintenance',
        'risk': 'Low',
        'symptoms': ['None - preventive care'],
        'tests': ['Annual physical exam', 'Dental checkup', 'Vision screening'],
        'actions': [
            'Annual influenza vaccination',
            'Age-appropriate cancer screenings',
            'Regular dental cleanings every 6 months',
            'Mental health check-in annually'
        ],
        'severity': 'info',
        'icon': '🛡️'
    })
    
    # Sort by severity (critical first, then warning, then info)
    severity_order = {'critical': 0, 'warning': 1, 'info': 2}
    recommendations.sort(key=lambda x: severity_order[x['severity']])
    
    return recommendations

# ============================================
# ENHANCED MODEL LOADING WITH CACHING
# ============================================
@st.cache_resource(show_spinner="Loading and training advanced model...")
def get_enhanced_model():
    """
    Load or train an enhanced model with feature engineering
    """
    model_path = 'models/diabetes_advanced_model.pkl'
    feature_names_path = 'models/feature_names.pkl'
    
    # Check if enhanced model exists
    if os.path.exists(model_path) and os.path.exists(feature_names_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
        st.success("✅ Loaded pre-trained advanced model")
        return model, feature_names
    
    # Load and preprocess data with advanced feature engineering
    if os.path.exists('data/diabetes.csv'):
        df = pd.read_csv('data/diabetes.csv')
        
        # Enhanced preprocessing
        # Handle missing values (0s in this dataset often mean missing)
        for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
            df[col] = df[col].replace(0, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
        
        # ADVANCED FEATURE ENGINEERING
        # Create new informative features
        df['Glucose_BMI_Ratio'] = df['Glucose'] / df['BMI']
        df['BloodPressure_Age_Product'] = df['BloodPressure'] * df['Age']
        df['Insulin_Glucose_Ratio'] = df['Insulin'] / df['Glucose'].replace(0, 1)
        df['Metabolic_Age'] = df['BMI'] * df['Age'] / 100
        
        # Create risk factor scores
        df['Glucose_Risk'] = pd.cut(df['Glucose'], 
                                   bins=[0, 100, 125, 200, 300],
                                   labels=[0, 1, 2, 3]).astype(int)
        
        df['BMI_Risk'] = pd.cut(df['BMI'],
                               bins=[0, 18.5, 25, 30, 100],
                               labels=[0, 1, 2, 3]).astype(int)
        
        # Prepare features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Train advanced model with more trees and features
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with progress indicator
        model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model and feature names
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(list(X.columns), feature_names_path)
        
        st.success(f"✅ Trained new advanced model with {accuracy:.2%} accuracy")
        return model, list(X.columns)
    
    # Fallback to basic model
    st.warning("⚠️ Using basic model (dataset not found)")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# ============================================
# ENHANCED PREDICTION FUNCTION
# ============================================
def get_detailed_prediction(model, input_data, feature_names):
    """
    Get prediction with confidence scores and explanations
    """
    try:
        # Create DataFrame with all expected features
        input_df = pd.DataFrame([input_data])
        
        # Add engineered features
        input_df['Glucose_BMI_Ratio'] = input_df['Glucose'] / input_df['BMI'].replace(0, 1)
        input_df['BloodPressure_Age_Product'] = input_df['BloodPressure'] * input_df['Age']
        input_df['Insulin_Glucose_Ratio'] = input_df['Insulin'] / input_df['Glucose'].replace(0, 1)
        input_df['Metabolic_Age'] = input_df['BMI'] * input_df['Age'] / 100
        
        # Add risk scores
        input_df['Glucose_Risk'] = pd.cut(input_df['Glucose'],
                                        bins=[0, 100, 125, 200, 300],
                                        labels=[0, 1, 2, 3]).astype(int)
        
        input_df['BMI_Risk'] = pd.cut(input_df['BMI'],
                                     bins=[0, 18.5, 25, 30, 100],
                                     labels=[0, 1, 2, 3]).astype(int)
        
        # Align columns with training data
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_names]
        
        # Get prediction and probabilities
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100
        
        # Get feature importance for this specific prediction
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features_idx = np.argsort(importances)[-5:][::-1]
            top_features = [feature_names[i] for i in top_features_idx]
            top_importances = [importances[i] for i in top_features_idx]
        else:
            top_features = []
            top_importances = []
        
        # Get disease-specific recommendations
        disease_recommendations = get_disease_recommendations(input_data)
        
        return {
            'prediction': int(prediction),
            'probability': float(probabilities[1]),
            'confidence': confidence,
            'top_features': top_features,
            'top_importances': top_importances,
            'disease_recommendations': disease_recommendations,
            'input_parameters': input_data
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_risk_gauge(probability):
    """Create an interactive gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Diabetes Risk Score", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#10B981'},
                {'range': [30, 70], 'color': '#F59E0B'},
                {'range': [70, 100], 'color': '#EF4444'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def create_feature_importance_chart(features, importances):
    """Create horizontal bar chart for feature importance"""
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color='#2E86AB',
            text=[f'{imp:.1%}' for imp in importances],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top Contributing Factors",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def create_disease_risk_chart(recommendations):
    """Create a bar chart showing disease risks"""
    diseases = [rec['disease'] for rec in recommendations]
    risks = []
    
    for rec in recommendations:
        if rec['risk'] == 'High':
            risks.append(3)
        elif rec['risk'] == 'Moderate':
            risks.append(2)
        else:
            risks.append(1)
    
    fig = go.Figure(data=[
        go.Bar(
            x=diseases,
            y=risks,
            marker_color=['#EF4444' if r == 3 else '#F59E0B' if r == 2 else '#2E86AB' for r in risks],
            text=[rec['icon'] + ' ' + rec['risk'] for rec in recommendations],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Disease Risk Assessment",
        yaxis_title="Risk Level",
        xaxis_title="Condition",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Moderate', 'High']
        )
    )
    return fig

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # Title and description
    st.markdown('<h1 class="main-header">🏥 Advanced Diabetes Risk Prediction</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #666;'>
        A machine learning system using <strong>Random Forest Classifier</strong> trained on 
        <strong>Pima Indians Diabetes Dataset</strong> with advanced feature engineering and 
        <strong>disease-specific recommendations</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load enhanced model
    with st.spinner("🔄 Initializing advanced prediction system..."):
        model, feature_names = get_enhanced_model()
    
    # ============================================
    # ENHANCED SIDEBAR WITH TABS
    # ============================================
    with st.sidebar:
        st.markdown("## ⚙️ Patient Configuration")
        
        # Create tabs in sidebar
        tab1, tab2, tab3 = st.tabs(["Clinical", "Lifestyle", "Advanced"])
        
        with tab1:
            st.markdown("#### Clinical Parameters")
            col1, col2 = st.columns(2)
            with col1:
                pregnancies = st.slider("Pregnancies", 0, 20, 1, 
                                       help="Number of times pregnant")
                glucose = st.slider("Glucose (mg/dL)", 50, 300, 120,
                                   help="Plasma glucose concentration")
                blood_pressure = st.slider("Blood Pressure", 40, 150, 80,
                                          help="Diastolic blood pressure (mm Hg)")
            with col2:
                skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 25,
                                          help="Triceps skin fold thickness")
                insulin = st.slider("Insulin (μU/mL)", 0, 900, 100,
                                   help="2-Hour serum insulin")
                bmi = st.slider("BMI", 10.0, 70.0, 25.0, 0.1,
                               help="Body Mass Index (weight in kg/(height in m)²)")
        
        with tab2:
            st.markdown("#### Lifestyle & History")
            dpf = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.5, 0.01,
                           help="Diabetes likelihood based on family history")
            age = st.slider("Age", 20, 100, 35,
                           help="Age in years")
            
            # Additional lifestyle factors
            st.markdown("---")
            exercise = st.select_slider(
                "Weekly Exercise",
                options=["None", "Light", "Moderate", "Heavy", "Athlete"],
                value="Moderate"
            )
            
            diet = st.select_slider(
                "Diet Quality",
                options=["Poor", "Average", "Good", "Excellent"],
                value="Average"
            )
        
        with tab3:
            st.markdown("#### Advanced Settings")
            model_confidence = st.slider(
                "Model Confidence Threshold", 50, 95, 70,
                help="Minimum confidence required for high-risk classification"
            )
            
            show_technical = st.checkbox(
                "Show Technical Details",
                value=False,
                help="Display model internals and metrics"
            )
            
            show_disease_details = st.checkbox(
                "Show Detailed Disease Analysis",
                value=True,
                help="Display comprehensive disease recommendations"
            )
        
        # Prediction button with icon
        predict_btn = st.button(
            "🔍 Predict Diabetes Risk",
            type="primary",
            use_container_width=True
        )
        
        # Quick actions
        st.markdown("---")
        st.markdown("#### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("📊 View Data", use_container_width=True):
                st.session_state.show_data = True
    
    # ============================================
    # MAIN CONTENT AREA
    # ============================================
    if predict_btn:
        # Prepare input data
        input_data = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        # Add progress animation
        with st.spinner("🤖 Analyzing health parameters with AI..."):
            time.sleep(1)  # Simulate processing time
            result = get_detailed_prediction(model, input_data, feature_names)
        
        if result:
            # ============================================
            # PREDICTION RESULTS SECTION
            # ============================================
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', 
                       unsafe_allow_html=True)
            
            # Create two columns for results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Risk Gauge
                gauge_fig = create_risk_gauge(result['probability'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Risk Level Card
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                if result['probability'] < 0.3:
                    risk_level = "Low Risk"
                    risk_class = "risk-low"
                    icon = "✅"
                    recommendation = "Maintain healthy lifestyle with annual checkups."
                elif result['probability'] < 0.7:
                    risk_level = "Moderate Risk"
                    risk_class = "risk-medium"
                    icon = "⚠️"
                    recommendation = "Consult doctor. Monitor glucose levels regularly."
                else:
                    risk_level = "High Risk"
                    risk_class = "risk-high"
                    icon = "🚨"
                    recommendation = "Immediate medical consultation required."
                
                st.markdown(f"### {icon} {risk_level}")
                st.markdown(f'<div class="{risk_class}">{risk_level}</div>', 
                           unsafe_allow_html=True)
                
                st.metric(
                    "Probability",
                    f"{result['probability']:.1%}",
                    delta=f"{result['confidence']:.1f}% confidence",
                    delta_color="inverse" if result['prediction'] == 1 else "normal"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ============================================
            # DISEASE-SPECIFIC RECOMMENDATIONS SECTION
            # ============================================
            st.markdown('<h2 class="sub-header">🩺 Disease-Specific Recommendations</h2>', 
                       unsafe_allow_html=True)
            
            st.info("""
            💡 **Based on your clinical parameters, here are specific conditions to monitor 
            and actionable recommendations for each. Conditions are prioritized by severity.**
            """)
            
            # Disease risk visualization
            disease_chart = create_disease_risk_chart(result['disease_recommendations'])
            st.plotly_chart(disease_chart, use_container_width=True)
            
            # Display detailed disease cards
            for i, disease in enumerate(result['disease_recommendations']):
                # Determine CSS class based on severity
                if disease['severity'] == 'critical':
                    css_class = "disease-critical"
                    severity_icon = "🔴"
                elif disease['severity'] == 'warning':
                    css_class = "disease-warning"
                    severity_icon = "🟡"
                else:
                    css_class = "disease-info"
                    severity_icon = "🔵"
                
                # Create disease card
                with st.container():
                    st.markdown(f'<div class="disease-card {css_class}">', unsafe_allow_html=True)
                    
                    # Header with icon and disease name
                    col1, col2, col3 = st.columns([1, 4, 2])
                    with col1:
                        st.markdown(f"### {disease['icon']}")
                    with col2:
                        st.markdown(f"**{disease['disease']}**")
                    with col3:
                        st.markdown(f"{severity_icon} **{disease['risk']} Risk**")
                    
                    # Symptoms
                    with st.expander("🔍 Common Symptoms", expanded=False):
                        for symptom in disease['symptoms']:
                            st.markdown(f"• {symptom}")
                    
                    # Recommended Tests
                    with st.expander("🧪 Recommended Diagnostic Tests", expanded=False):
                        for test in disease['tests']:
                            st.markdown(f"• {test}")
                    
                    # Action Steps (always expanded for critical conditions)
                    expand_action = disease['severity'] == 'critical'
                    with st.expander("🚨 Immediate Action Steps", expanded=expand_action):
                        for action in disease['actions']:
                            st.markdown(f"• {action}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # ============================================
            # HEALTH INSIGHTS SECTION
            # ============================================
            st.markdown('<h2 class="sub-header">🔍 Health Insights</h2>', 
                       unsafe_allow_html=True)
            
            # Create insights cards
            insights_col1, insights_col2, insights_col3 = st.columns(3)
            
            with insights_col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 📈 Key Risk Factors")
                if glucose > 140:
                    st.error("⚠️ Elevated glucose levels")
                if bmi > 30:
                    st.warning("⚠️ BMI indicates obesity risk")
                if age > 45:
                    st.info("ℹ️ Age is a contributing factor")
                if dpf > 0.8:
                    st.error("⚠️ Strong family history")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with insights_col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Parameter Analysis")
                
                # Feature importance visualization
                if result['top_features']:
                    imp_fig = create_feature_importance_chart(
                        result['top_features'][:3],
                        result['top_importances'][:3]
                    )
                    st.plotly_chart(imp_fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with insights_col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("#### 💡 General Recommendations")
                st.success(recommendation)
                
                # Additional recommendations based on parameters
                if exercise == "None":
                    st.info("🏃 Start light exercise routine")
                if diet == "Poor":
                    st.info("🥗 Improve dietary habits")
                if bmi > 25:
                    st.info("⚖️ Consider weight management")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ============================================
            # TECHNICAL DETAILS (COLLAPSIBLE)
            # ============================================
            with st.expander("🔬 View Technical Details", expanded=False):
                tab1, tab2, tab3 = st.tabs(["Parameters", "Model Info", "Data"])
                
                with tab1:
                    st.markdown("#### Input Parameters")
                    param_data = pd.DataFrame({
                        'Parameter': ['Glucose', 'BMI', 'Age', 'Blood Pressure', 
                                     'Insulin', 'Pregnancies', 'Diabetes Pedigree', 
                                     'Skin Thickness'],
                        'Value': [glucose, bmi, age, blood_pressure, insulin, 
                                 pregnancies, dpf, skin_thickness],
                        'Normal Range': ['70-100', '18.5-25', '<45', '<120', 
                                        '<140', '<5', '<0.5', '<30'],
                        'Status': [
                            'Normal' if 70 <= glucose <= 100 else 'High' if glucose > 100 else 'Low',
                            'Normal' if 18.5 <= bmi <= 25 else 'High' if bmi > 25 else 'Low',
                            'Normal' if age <= 45 else 'High',
                            'Normal' if blood_pressure <= 120 else 'High',
                            'Normal' if insulin <= 140 else 'High',
                            'Normal' if pregnancies <= 5 else 'High',
                            'Normal' if dpf <= 0.5 else 'High',
                            'Normal' if skin_thickness <= 30 else 'High'
                        ]
                    })
                    st.dataframe(param_data, use_container_width=True, hide_index=True)
                
                with tab2:
                    st.markdown("#### Model Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Type", "Random Forest")
                        st.metric("Ensemble Size", "300 Trees")
                        st.metric("Feature Engineering", "Enabled")
                    with col2:
                        st.metric("Training Samples", "768")
                        st.metric("Features Used", len(feature_names))
                        st.metric("Prediction Confidence", f"{result['confidence']:.1f}%")
                
                with tab3:
                    st.markdown("#### Dataset Statistics")
                    if os.path.exists('data/diabetes.csv'):
                        df = pd.read_csv('data/diabetes.csv')
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Diabetic Cases", f"{df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
                        with col3:
                            st.metric("Features", len(df.columns) - 1)
                        
                        if st.checkbox("Show Sample Data"):
                            st.dataframe(df.head(10), use_container_width=True)
    
    else:
        # ============================================
        # DASHBOARD VIEW (When no prediction made)
        # ============================================
        st.markdown("---")
        
        # Welcome message
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 2rem 0;'>
            <h2 style='color: white;'>Welcome to Advanced Diabetes Prediction</h2>
            <p style='font-size: 1.1rem;'>
                Configure patient parameters in the sidebar and click <strong>"Predict Diabetes Risk"</strong><br>
                to get AI-powered risk assessment with <strong>disease-specific recommendations</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🏥 Dataset")
            st.markdown("**Pima Indians**")
            st.markdown("768 patient records")
            st.markdown("9 clinical features")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🤖 AI Model")
            st.markdown("Random Forest")
            st.markdown("300 decision trees")
            st.markdown("Advanced features")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("#### 🩺 Disease Analysis")
            st.markdown("Condition-specific")
            st.markdown("Risk-stratified")
            st.markdown("Actionable steps")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 3rem;'>
        <p>
            <strong>Advanced Diabetes Risk Prediction System</strong> | 
            Machine Learning Project with Disease Recommendations | 
            For educational and demonstration purposes
        </p>
        <p style='margin-top: 1rem;'>
            ⚠️ <em>This tool is for screening purposes only. Always consult healthcare professionals for medical decisions.</em>
        </p>
        <p style='margin-top: 0.5rem; font-size: 0.8rem;'>
            Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    main()