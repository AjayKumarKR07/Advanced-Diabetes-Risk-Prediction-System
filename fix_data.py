import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

print("Generating clean diabetes dataset...")

data = {
    'Pregnancies': np.random.randint(0, 17, n_samples),
    'Glucose': np.random.randint(0, 199, n_samples),
    'BloodPressure': np.random.randint(0, 122, n_samples),
    'SkinThickness': np.random.randint(0, 99, n_samples),
    'Insulin': np.random.randint(0, 846, n_samples),
    'BMI': np.random.uniform(0, 67.1, n_samples),
    'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
    'Age': np.random.randint(21, 81, n_samples),
}

def calculate_diabetes_prob(row):
    prob = 0
    prob += row['Glucose'] * 0.003
    prob += row['BMI'] * 0.01
    prob += row['Age'] * 0.005
    prob += row['DiabetesPedigreeFunction'] * 0.1
    prob += row['Pregnancies'] * 0.02
    return 1 if prob > 0.5 else 0

df = pd.DataFrame(data)
df['Outcome'] = df.apply(calculate_diabetes_prob, axis=1)

# Add some missing values (0s in this dataset often mean missing)
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    df.loc[df.sample(frac=0.05).index, col] = 0

# Save to CSV WITHOUT index and with proper formatting
df.to_csv('data/diabetes.csv', index=False)

print(f"✅ Clean dataset created with shape: {df.shape}")
print(f"✅ Diabetic cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
print(f"✅ First few rows:")
print(df.head())