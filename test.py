import streamlit as st
import joblib
import numpy as np

# ── Load models ──
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
scaler = joblib.load("minmax_scaler.pkl")
model = joblib.load("logistic_regression_model.pkl")

st.title("Heart Disease Predictor")

# ── Inputs to scale ──
bmi = st.number_input("BMI", min_value=0.0)
physical_health = st.number_input("Physical Health", min_value=0.0)
mental_health = st.number_input("Mental Health", min_value=0.0)
sleep_time = st.number_input("Sleep Time", min_value=0.0)

# ── Checkboxes ──
smoking = st.checkbox("Smoking")
alcohol_drinking = st.checkbox("Alcohol Drinking")
stroke = st.checkbox("Stroke")
diff_walking = st.checkbox("Difficulty Walking")
physical_activity = st.checkbox("Physical Activity")
asthma = st.checkbox("Asthma")
kidney_disease = st.checkbox("Kidney Disease")

# ── Ordinal Encoded Inputs ──
diabetic = st.selectbox("Diabetic", ['Yes', 'Yes (during pregnancy)', 'No, borderline diabetes', 'No'])
gen_health = st.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
age_cat = st.selectbox("Age Category", [
    '18-24', '25-29', '30-34', '35-39', '40-44',
    '45-49', '50-54', '55-59', '60-64',
    '65-69', '70-74', '75-79', '80 or older'
])

if st.button("Predict"):
    # Encode 3 values
    gen_enc, diab_enc, age_enc = ordinal_encoder.transform([[gen_health, diabetic, age_cat]])[0]

    # Scale numeric values
    to_scale = np.array([[bmi, sleep_time, age_enc, physical_health, mental_health]])
    scaled = scaler.transform(to_scale)[0]

    # Binary checkbox inputs
    binary = [
        int(smoking), int(alcohol_drinking), int(stroke), int(diff_walking),
        int(physical_activity), int(asthma), int(kidney_disease)
    ]

    # Final input: 5 scaled + 7 binary + 3 encoded = 15
    x = np.array(list(scaled) + binary + [gen_enc, diab_enc, age_enc]).reshape(1, -1)

    # Predict
    prediction = model.predict(x)[0]
    proba = model.predict_proba(x)[0][1]  # probability of class 1

    result = "🔴 Heart Disease: **True**" if prediction == 1 else "🟢 Heart Disease: **False**"
    st.success(result)
    st.write(f"Confidence: **{proba * 100:.2f}%** chance of heart disease")
