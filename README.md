# 🩺 Advanced Diabetes Risk Prediction System

A full-stack Machine Learning system that predicts diabetes risk using clinical and lifestyle parameters.  
This project includes data preprocessing, feature engineering, model training, Streamlit deployment, and Kaggle leaderboard submission.


## 🚀 Live Demo
🔗 **Streamlit App:**  
https://advanced-diabetes-risk-prediction-system-sqgjemwis3jbvzspq7vb8.streamlit.app/


## 🏆 Kaggle Competition
**Playground Series S5E12 – Diabetes Prediction**  
Public Leaderboard ROC-AUC: **0.68538**


## 📊 Features
- Advanced Random Forest ML model
- Categorical feature encoding & missing value handling
- Feature engineering for improved accuracy
- Interactive Streamlit web interface
- Kaggle competition submission pipeline
- ROC-AUC, confusion matrix & evaluation charts


## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-Learn
- Streamlit
- Matplotlib & Seaborn
- Kaggle API


## 📁 Project Structure

Diabetes-prediction-project/
│
├── data/ # Datasets (train.csv, test.csv, etc.)
├── models/ # Trained ML models
├── notebooks/ # Experiments & EDA
├── results/ # Graphs & reports
├── src/ # Source code
├── tests/ # Unit tests
├── train_kaggle_model.py # Kaggle training pipeline
├── kaggle_submit.py # Kaggle submission generator
├── main.py # Streamlit app
├── requirements.txt
└── README.md


## ⚙️ Installation

```bash
git clone https://github.com/AjayKumarKR07/Advanced-Diabetes-Risk-Prediction-System.git
cd Diabetes-prediction-project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
▶️ Run Web App
bash
Copy code
streamlit run main.py
📤 Kaggle Submission
bash
Copy code
python train_kaggle_model.py
python kaggle_submit.py
Upload the generated submission.csv to Kaggle.

📈 Results
Validation ROC-AUC: 0.70+

Kaggle Public Score: 0.68538

📌 Author
Ajay Kumar KR
GitHub: https://github.com/AjayKumarKR07

