import streamlit as st
import joblib
import numpy as np


@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_model.pkl')

model = load_model()

st.title("Breast Cancer Prediction (Random Forest)")
st.write("Enter the clinical measurements to predict diagnosis.")


col1, col2 = st.columns(2)

with col1:
    mean_radius = st.slider("Mean Radius", 5.0, 30.0, 15.0)
    mean_texture = st.slider("Mean Texture", 5.0, 40.0, 20.0)
    mean_perimeter = st.slider("Mean Perimeter", 40.0, 200.0, 90.0)

with col2:
    mean_area = st.slider("Mean Area", 100.0, 2500.0, 500.0)
    mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.2, 0.1)

if st.button("Predict"):
  
    features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]])
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")
 
