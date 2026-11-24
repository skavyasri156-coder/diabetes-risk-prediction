import pandas as pd
import numpy as np

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

df = pd.read_csv("dataset/pima-indians-diabetes.csv", header=None, names=columns)

# 1. In these columns, 0 is impossible, so treat 0 as "missing"
cols_with_fake_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_fake_zeros] = df[cols_with_fake_zeros].replace(0, np.nan)

print("Missing values after replacing zeros:")
print(df.isnull().sum())
print()

# 2. Fill missing values with the MEDIAN of each column
for col in cols_with_fake_zeros:
    df[col] = df[col].fillna(df[col].median())

print("Missing values after filling:")
print(df.isnull().sum().sum(), "missing values left")
print()

# 3. Save the cleaned data as a new file
df.to_csv("dataset/diabetes_clean.csv", index=False)
print("Saved: dataset/diabetes_clean.csv")
print(df.describe().round(1))