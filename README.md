# Diabetes Risk Predictor

**Live Demo:** (add your Streamlit link here after deploying)

An educational machine learning web app that estimates diabetes risk from
health measurements, built with Python, scikit-learn and Streamlit.

> **Disclaimer:** Educational project only. NOT a medical diagnosis tool.

## Dataset
Pima Indians Diabetes dataset (768 records, 8 features). Zero values in
Glucose, BloodPressure, SkinThickness, Insulin and BMI were treated as
missing and filled with the column median.

## Model
- Random Forest (200 trees, max_depth 6, class_weight="balanced")
- 5-fold cross-validation: accuracy about 75.5%, recall about 76.9%, ROC-AUC about 0.838
- Data is imbalanced (500 vs 268), so recall was also evaluated.
  Balancing raised recall from about 57% to about 77%.

## Limitations
Small dataset; unusual feature combinations can give unreliable results.

## How to Run
pip install -r requirements.txt
python -m streamlit run app.py

## Author
S. Kavyasri, B.Tech CSE (Data Science)