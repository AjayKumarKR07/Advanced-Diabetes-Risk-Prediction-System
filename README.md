# 🩺 Advanced Diabetes Risk Prediction System

A full-stack Machine Learning system that predicts the probability of diabetes using clinical and lifestyle parameters.  
This project includes data preprocessing, feature engineering, model training, Streamlit deployment, and Kaggle competition submission.



## 🚀 Live Web Application
🔗 https://advanced-diabetes-risk-prediction-system-sqgjemwis3jbvzspq7vb8.streamlit.app/



## 🏆 Kaggle Competition
**Playground Series S5E12 – Diabetes Prediction**  
Public Leaderboard ROC-AUC: **0.68538**



## 📌 Project Objectives
- Predict diabetes risk with high accuracy  
- Provide real-time screening through a web interface  
- Build a complete Kaggle ML pipeline  
- Apply feature engineering and categorical encoding  
- Evaluate model performance using ROC-AUC  



## ✨ Key Features
- Random Forest classification model  
- Categorical data encoding & missing value handling  
- Feature engineering for improved accuracy  
- Interactive Streamlit web interface  
- Kaggle submission automation  
- Real-time prediction & probability output  



## 🏗 System Architecture

User Input / Dataset
↓
Data Cleaning & Preprocessing
↓
Feature Engineering
↓
Random Forest ML Model
↓
Probability Prediction
↓
Risk Classification Dashboard


## 🧠 Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Core Programming |
| Pandas, NumPy | Data Handling |
| Scikit-Learn | Machine Learning |
| Random Forest | Prediction Model |
| Streamlit | Web Interface |
| Matplotlib & Seaborn | Visualization |
| Kaggle API | Competition Submission |


## 📁 Project Structure

Diabetes-prediction-project/
│
├── data/ # Kaggle datasets
├── models/ # Trained ML models
├── notebooks/ # Experiments & EDA
├── results/ # Graphs & reports
├── src/ # Source code
├── tests/ # Unit tests
├── train_kaggle_model.py # Kaggle training pipeline
├── kaggle_submit.py # Kaggle submission generator
├── main.py # Streamlit application
├── requirements.txt
└── README.md


## ⚙️ Installation

```bash
git clone https://github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System.git
cd Diabetes-prediction-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
▶️ Run the Web Application
bash
Copy code
streamlit run main.py
📤 Kaggle Submission Workflow
bash
Copy code
python train_kaggle_model.py
python kaggle_submit.py
Upload the generated submission.csv to Kaggle.

📈 Results
Validation ROC-AUC: 0.70+

Kaggle Public Score: 0.68538

👨‍💻 Author
Ajay Kumar KR
GitHub: https://github.com/AjayKumarKR07

