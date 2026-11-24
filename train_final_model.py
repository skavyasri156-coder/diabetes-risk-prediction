import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/diabetes_clean.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = RandomForestClassifier(
    n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
)
model.fit(X, y)

joblib.dump(model, "diabetes_model.pkl")

print("Model trained on", len(df), "records")
print("Saved as diabetes_model.pkl")
print("Feature order:", list(X.columns))