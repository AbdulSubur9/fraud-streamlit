import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load trained model
model = joblib.load('model.pkl')

st.title("💳 Credit Card Fraud Detection")
st.markdown("This app predicts the likelihood of a transaction being fraudulent.")

# === User Inputs ===
amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)

hour = st.slider("Hour of Transaction (0 - 23)", 0, 23)

transaction_type = st.selectbox("Transaction Type", ["POS", "Online", "ATM"])

# One-hot encoding for type (simulated input for simplicity)
transaction_map = {
    "POS": [1, 0, 0],
    "Online": [0, 1, 0],
    "ATM": [0, 0, 1]
}

if st.button("Check for Fraud"):
    features = np.zeros((1, 29))

    # Fill in the values
    features[0, -1] = amount   # Amount
    features[0, 0] = hour      # Simulating 'V1' as Hour
    features[0, 1:4] = transaction_map[transaction_type]  # Simulating V2-V4 for type

    # Predict
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][1] * 100  # Probability of fraud

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! Confidence: {confidence:.2f}%")
    else:
        st.success(f"✅ Safe Transaction. Fraud Risk: {confidence:.2f}%")
