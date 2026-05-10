import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
st.set_page_config(
    page_title="Muti Disease Prediction",
    page_icon="🏥",
    layout="wide"
)
st.title("🩺 Advanced Multi Disease Prediction System")

st.write("AI-powered healthcare prediction system using Machine Learning")
selected=option_menu(
    menu_title=None,
    options=["Home","Diabetes","Heart Disease","History"],
    icons=["house","activity","heart","clock-history"],
    orientation="horizontal"
)
# Load models
model_d = pickle.load(open("diabetes.pkl", "rb"))
scaler_d = pickle.load(open("diabetes_scaler.pkl", "rb"))

model_h = pickle.load(open("heart.pkl", "rb"))
scaler_h = pickle.load(open("heart_scaler.pkl", "rb"))


# Sidebar
disease = st.sidebar.selectbox("🔍 Select Disease", ["Diabetes", "Heart Disease"])


# ---------------- DIABETES ----------------

if disease == "Diabetes":

    st.subheader("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        glu = st.number_input("Glucose")
        ratio = st.number_input("Ratio")
        age = st.number_input("Age")

    with col2:
        hdl = st.number_input("HDL")
        weight = st.number_input("Weight")

    if st.button("Predict Diabetes"):

        data = np.array([[glu, hdl, ratio, age, weight]])
        data_scaled = scaler_d.transform(data)

        prediction = model_d.predict(data_scaled)
        prob = model_d.predict_proba(data_scaled)

        if prediction[0] == 1:
            st.error(f"Diabetes Detected ❗ (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"No Diabetes ✅ (Confidence: {prob[0][0]*100:.2f}%)")


# ---------------- HEART ----------------

elif disease == "Heart Disease":

    st.subheader("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age")
        cp = st.number_input("Chest Pain")

    with col2:
        sex = st.number_input("Sex")
        thalach = st.number_input("Heart Rate")

    if st.button("Predict Heart Disease"):

        data = np.array([[age, sex, cp, thalach]])
        data_scaled = scaler_h.transform(data)

        prediction = model_h.predict(data_scaled)
        prob = model_h.predict_proba(data_scaled)

        if prediction[0] == 1:
            st.error(f"Heart Disease Detected ❗ (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"No Heart Disease ✅ (Confidence: {prob[0][0]*100:.2f}%)")


st.markdown("---")
st.write("Developed by Yashraj")
st.markdown("---")
st.write("Developed by Yashraj")
