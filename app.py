import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Page Setup
st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")

# 2. Load the Model
@st.cache_resource
def load_model():
    # Ensure this matches your uploaded file name
    return joblib.load('breast_cancer_model.pkl')

model = load_model()

st.title("🩺 Breast Cancer Detection System")
st.write("Provide the 20 clinical measurements below to get a diagnostic prediction.")

# 3. Define your 20 features 
# REPLACE THESE NAMES with your actual column names from Colab
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error"
]

# 4. Create UI Input Fields (Organized into 4 columns)
st.subheader("Input Clinical Parameters")
inputs = []
cols = st.columns(4)

for i, name in enumerate(feature_names):
    with cols[i % 4]:
        # You can adjust the min_value, max_value, and value (default) based on your data
        val = st.number_input(f"{name}", value=0.0, format="%.4f")
        inputs.append(val)

st.markdown("---")

# 5. Prediction Logic
if st.button("Generate Diagnostic Report", type="primary"):
    # Convert inputs to a 2D array (1 row, 20 columns)
    input_array = np.array([inputs])
    
    # Predict
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array) # Probability of the classes

    # 6. Display Results
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.subheader("Result")
        if prediction[0] == 1:
            st.error("🚨 Prediction: MALIGNANT")
        else:
            st.success("✅ Prediction: BENIGN")
            
    with col_res2:
        st.subheader("Confidence")
        confidence = prediction_proba[0][prediction[0]]
        st.write(f"The model is **{confidence:.2%}** confident in this result.")

st.info("**Disclaimer:** This is an AI-assisted tool for educational research. It is not a substitute for professional medical advice.")
    
