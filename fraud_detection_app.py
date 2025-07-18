import streamlit as st
import joblib
import json
import pandas as pd

# Load model, default values, and threshold
model = joblib.load("fraud_risk_pipeline.pkl")
default_values = joblib.load("median_values.pkl")
with open("threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

st.title("Credit Card Fraud Detection App")
st.markdown("This model is trained on the Kaggle credit card dataset using PCA-transformed features. Input values are standardized components.")

st.header("Enter Transaction Details")

# Full 30 features used in model (in order)
all_features = list(default_values.keys())

# Top 20 SHAP features to show in UI
top_features = [
    'V3', 'V17', 'V12', 'V10', 'V14', 'V7', 'V18', 'V6', 'V16', 'V11',
    'V5', 'V2', 'V4', 'V1', 'V28', 'V9', 'V21', 'V19', 'V26', 'Amount'
]

# Features to auto-fill with defaults (not shown in form)
remaining_features = [f for f in all_features if f not in top_features]

# Get user input for top 20 features
input_data = {}
for feature in top_features:
    default = float(default_values[feature])
    step = 1.0 if feature == "Amount" else 0.01
    input_data[feature] = st.number_input(f"{feature}", value=default, step=step)

# Fill the remaining 10 features using default values
for feature in remaining_features:
    input_data[feature] = float(default_values[feature])

# Predict on click
if st.button("Predict Fraud Risk"):
    input_df = pd.DataFrame([input_data])[all_features]  # Ensure column order
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if probability >= threshold:
        st.error(f"**Fraudulent Transaction Detected**  \nProbability: {probability:.2f}")
    else:
	    st.success(f"**Legitimate transaction**  \nProbability: {probability:.2f}")
     
