import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/diabetes_clean.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression (balanced)": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "Random Forest (balanced)": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    ),
}

for name, model in models.items():
    s = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    print(name)
    print("  Accuracy:", round(s["test_accuracy"].mean() * 100, 2), "%")
    print("  Recall  :", round(s["test_recall"].mean() * 100, 2), "%")
    print("  ROC-AUC :", round(s["test_roc_auc"].mean(), 3))
    print()