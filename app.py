import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
    
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
if "history" not in st.session_state:
    st.session_state.history=[]
# Load models
model_d = pickle.load(open("diabetes.pkl", "rb"))
scaler_d = pickle.load(open("diabetes_scaler.pkl", "rb"))

model_h = pickle.load(open("heart.pkl", "rb"))
scaler_h = pickle.load(open("heart_scaler.pkl", "rb"))


if selected== "Home":
    st.header("Welcome to Advanced Multi Disease Prediction System")
    st.write("""
    This AI-powered healthcare application predicts diseases using Machine Learning.

    Features:
    -Diabetes Prediction 
    -Heart Diseases Prediction
    -Risk Visualization
    -Live Web Deployement
    """)
    st.success("System Running Sucessfully")
    


# ---------------- DIABETES ----------------

elif selected == "Diabetes":

    st.subheader("Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        glu = st.number_input("Glucose Level (stab.glu)", min_value=0.0, value=100.0)
        ratio = st.number_input("Cholesterol Ratio", min_value=0.0, value=4.0)
        age = st.number_input("Age", min_value=1, value=40,step=1)

    with col2:
        hdl = st.number_input("HDL Cholesterol", min_value=0.0, value=50.0)
        weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0)


    if st.button("Predict Diabetes"):
        data = np.array([[glu, hdl, ratio, age, weight]])
        data_scaled = scaler_d.transform(data)

        prediction = model_d.predict(data_scaled)
        prob = model_d.predict_proba(data_scaled)

        risk = prob[0][1] * 100
        st.session_state.history.append({
            "Diabetes":"Diabetes",
            "Risk":f"{risk:.2f}%"
        })

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={'text': "Diabetes Risk"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        st.plotly_chart(fig)
        st.progress(int(risk))
        st.write(f"Prediction Confidence: {risk:.2f}%")
        if prediction[0] == 1:
            st.error(f"Diabetes Detected ❗ (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"No Diabetes ✅ (Confidence: {prob[0][0]*100:.2f}%) ")



# ---------------- HEART ----------------

elif selected == "Heart Disease":

    st.subheader("Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, value=40)
        cp = st.selectbox("Chest Pain Type", [0,1,2,3])

    with col2:
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        thalach = st.number_input("Max Heart Rate", min_value=50, value=150)

    if st.button("Predict Heart Disease"):

        data = np.array([[age, sex, cp, thalach]])
        data_scaled = scaler_h.transform(data)

        prediction = model_h.predict(data_scaled)
        prob = model_h.predict_proba(data_scaled)
        risk=prob[0][1]*100
        st.session_state.history.append({
            "Heart Diseases":"Heart Disease",
            "Risk":f"{risk:.2f}%"
        })
        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text': "Heart Disease Risk"},
        gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig)
        st.progress(int(risk))
        st.write(f"Prediction Confidence: {risk:.2f}%")
        if prediction[0] == 1:
            st.error(f"Heart Disease Detected ❗ (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"No Heart Disease ✅ (Confidence: {prob[0][0]*100:.2f}%)")
elif selected=="history":
    st.header("Prediction History")
    if len(st.session_state.history)==0:
        st.warning("No Prediction yet")
    else:
        st.write(st.session_state.history)
     

st.markdown("---")
st.write("Developed by Yashraj")

