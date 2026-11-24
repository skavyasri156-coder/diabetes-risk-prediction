import pandas as pd

# 1. Column names (the CSV file has no header row)
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

# 2. Load the data
df = pd.read_csv("dataset/pima-indians-diabetes.csv", header=None, names=columns)

# 3. Look at the data
print("Shape (rows, columns):", df.shape)
print()
print(df.head())
print()
print("Outcome counts (0 = No diabetes, 1 = Diabetes):")
print(df["Outcome"].value_counts())
print()
print("Count of zero values in each column:")
print((df == 0).sum())