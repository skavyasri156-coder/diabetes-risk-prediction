import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")


@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")


model = load_model()

st.title("🩺 Diabetes Risk Predictor")
st.warning(
    "Educational project only. This is NOT a medical diagnosis tool. "
    "Please consult a doctor for any health concern."
)
st.write("Enter the details below and click **Predict**.")

# Input fields (same names and order as in training)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.slider("Glucose (mg/dL)", min_value=40, max_value=250, value=110)
blood_pressure = st.slider("Blood pressure (mm Hg)", min_value=20, max_value=130, value=70)
skin_thickness = st.slider("Skin thickness (mm)", min_value=5, max_value=100, value=25)
insulin = st.slider("Insulin (mu U/ml)", min_value=10, max_value=850, value=100)
bmi = st.slider("BMI", min_value=15.0, max_value=70.0, value=28.0, step=0.1)
dpf = st.slider("Diabetes pedigree function", min_value=0.05, max_value=2.5, value=0.4, step=0.01)
age = st.slider("Age (years)", min_value=18, max_value=90, value=30)

if st.button("Predict"):
    person = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
    )

    risk = model.predict_proba(person)[0][1] * 100

    if risk >= 60:
        st.error(f"High risk: {risk:.1f}% ❌")
    elif risk >= 35:
        st.warning(f"Moderate risk: {risk:.1f}% ⚠️")
    else:
        st.success(f"Low risk: {risk:.1f}% ✅")

    st.progress(int(risk))
    st.caption("Risk % is the model's estimate from training data, not a medical probability.")

st.caption("Model: Random Forest trained on the Pima Indians Diabetes dataset (768 records).")