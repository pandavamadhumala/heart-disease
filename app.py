import streamlit as st
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .stButton>button {
            background-color: #e63946;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #c1121f;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# Header
st.title("❤️ Heart Disease Risk Prediction System")
st.markdown("### Enter Patient Medical Details")
st.markdown("---")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 40)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120 (1 = Yes, 0 = No)", [0, 1])

with col2:
    restecg = st.number_input("Rest ECG (0-2)", 0, 2, 1)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.number_input("Slope (0-2)", 0, 2, 1)
    ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
    thal = st.number_input("Thal (0-3)", 0, 3, 1)

st.markdown("---")

if st.button("🔍 Predict Heart Disease Risk"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    if prediction[0] == 1:
        st.error("⚠ High Risk of Heart Disease")
        st.write(f"Confidence: {round(probability[0][1]*100, 2)}%")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write(f"Confidence: {round(probability[0][0]*100, 2)}%")

st.markdown("---")
st.caption("Model trained using RandomForestClassifier | Deployed with Streamlit")
