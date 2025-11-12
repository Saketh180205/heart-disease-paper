# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

st.title('Heart Disease Prediction Demo')

st.sidebar.header('Patient info')
age = st.sidebar.number_input('Age', 20, 100, 50)
sex = st.sidebar.selectbox('Sex (1=male,0=female)', [1,0], index=0)
cp = st.sidebar.slider('Chest pain type (cp) 0-3', 0, 3, 1)
trestbps = st.sidebar.number_input('Resting BP', 80, 240, 130)
chol = st.sidebar.number_input('Cholesterol', 100, 600, 240)
fbs = st.sidebar.selectbox('FastingBS > 120 mg/dl (1=yes)', [0,1], index=0)
restecg = st.sidebar.selectbox('Resting ECG (0-2)', [0,1,2], index=0)
thalach = st.sidebar.number_input('Max heart rate achieved', 60, 220, 150)
exang = st.sidebar.selectbox('Exercise induced angina (1=yes)', [0,1], index=0)
oldpeak = st.sidebar.number_input('ST depression (oldpeak)', 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox('Slope (0-2)', [0,1,2], index=1)
ca = st.sidebar.selectbox('Number of major vessels (0-4)', [0,1,2,3,4], index=0)
thal = st.sidebar.selectbox('thal (0-3)', [0,1,2,3], index=2)

input_data = np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1, -1)

try:
    model = joblib.load('results/logreg.joblib')
except Exception as e:
    st.error('Model not found. Run python src/train.py first and ensure results/logreg.joblib exists.')
    st.stop()

if st.button('Predict'):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0,1] if hasattr(model, 'predict_proba') else None
    if pred == 1:
        st.error(f'Affected by Defective Heart Disease — probability {prob:.2f}' if prob is not None else 'Affected')
    else:
        st.success(f'Healthy Heart — probability {1-prob:.2f}' if prob is not None else 'Healthy')
