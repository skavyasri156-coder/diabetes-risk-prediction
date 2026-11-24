import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("dataset/diabetes_clean.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Decision Tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42
    ),
}

print("5-fold cross-validation results")
print("-" * 55)
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    print(name)
    print("  Accuracy:", round(scores["test_accuracy"].mean() * 100, 2), "%")
    print("  Recall  :", round(scores["test_recall"].mean() * 100, 2), "%")
    print("  ROC-AUC :", round(scores["test_roc_auc"].mean(), 3))
    print()