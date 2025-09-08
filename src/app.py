import streamlit as st
import requests
import json
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="ChurnBuster Interactive Prediction",
    page_icon="⚡",
    layout="wide"
)

# --- API URL ---
# This is the address of your running FastAPI application
API_URL = "http://localhost:8001/predict"

# --- Page Title and Description ---
st.title("ChurnBuster ⚡")
st.markdown("Enter customer details below to get a real-time churn prediction.")

# --- Input Form ---
st.header("Customer Information")

# Create columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

# --- Predict Button and Logic ---
if st.button("Predict Churn", type="primary"):
    # 1. Create the payload dictionary from user inputs
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # 2. Call the FastAPI endpoint
    with st.spinner('Getting prediction...'):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            
            # 3. Display the result
            result = response.json()
            probability = result["churn_probability"]
            risk_band = result["risk_band"]

            st.success("Prediction successful!")
            
            # Display metrics
            st.metric(label="Churn Probability", value=f"{probability:.2%}")

            if risk_band == "High":
                st.error(f"Risk Band: {risk_band}")
                st.warning("This customer is at a high risk of churning. Consider targeted retention offers.")
            elif risk_band == "Medium":
                st.warning(f"Risk Band: {risk_band}")
            else:
                st.success(f"Risk Band: {risk_band}")

        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the prediction API. Please ensure it is running.")
            st.error(f"Details: {e}")