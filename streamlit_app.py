import streamlit as st
import pandas as pd
import joblib
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
# Set the page title, icon, and layout for a professional look
st.set_page_config(
    page_title="RetainIQ - Churn Prediction",
    page_icon="⚡",
    layout="wide"
)

# --- 2. LOAD MODELS AND EXPLAINER ---
# Use st.cache_resource to load these heavy objects only once, making the app faster.
@st.cache_resource
def load_model():
    """Loads the model, preprocessor, and creates the SHAP explainer."""
    try:
        model = joblib.load('models/xgb_model_v1.0.joblib')
        preprocessor = joblib.load('models/preprocessor_v1.0.joblib')
        explainer = shap.TreeExplainer(model)
        return model, preprocessor, explainer
    except FileNotFoundError:
        st.error("Model files not found! Please ensure 'models/xgb_model_v1.0.joblib' and 'models/preprocessor_v1.0.joblib' are in the repository.")
        return None, None, None

model, preprocessor, explainer = load_model()

# --- 3. APP HEADER ---
st.title("RetainIQ: Customer Churn Prediction & Explanation ⚡")
st.markdown("""
This application uses a machine learning model (XGBoost) to predict the likelihood of a customer churning. 
More importantly, it uses **SHAP (SHapley Additive exPlanations)** to explain the key factors that influence each prediction. 
Use the sidebar on the left to input customer details and see the results.
""")

# --- 4. USER INPUT SIDEBAR ---
with st.sidebar:
    st.header("Customer Information")
    
    # Create the input fields for all customer features
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen_str = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior_citizen = 1 if senior_citizen_str == "Yes" else 0 # Convert to 0/1 for the model
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=0.05)
    total_charges = st.slider("Total Charges ($)", min_value=18.0, max_value=9000.0, value=1000.0)

# --- 5. PREDICTION AND EXPLANATION LOGIC ---
# Only run the prediction if the model files were loaded successfully
if model and preprocessor and explainer:
    # Create a dictionary and then a DataFrame from the user's inputs
    feature_dict = {
        "gender": gender, "SeniorCitizen": senior_citizen, "Partner": partner, 
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "InternetService": internet_service,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    input_df = pd.DataFrame([feature_dict])

    # Preprocess the input data using the same pipeline from training
    processed_features = preprocessor.transform(input_df)
    
    # Get the prediction probability for the positive class (Churn=Yes)
    probability = model.predict_proba(processed_features)[0, 1]
    risk_band = "High" if probability > 0.5 else "Medium" if probability > 0.25 else "Low"

    # --- Display Prediction Result ---
    st.header("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Churn Probability", value=f"{probability:.2%}")
    with col2:
        # Display the risk band with color-coding
        if risk_band == "High":
            st.error(f"Risk Band: {risk_band}")
        elif risk_band == "Medium":
            st.warning(f"Risk Band: {risk_band}")
        else:
            st.success(f"Risk Band: {risk_band}")

    # --- Display SHAP Explanation ---
    st.header("Prediction Explanation")
    st.markdown("""
    The plot below is a **SHAP Force Plot**. It shows how each feature contributed to the final prediction.
    - **Base value:** The average churn probability across all customers.
    - **Features in <span style='color:red;'>red</span>** pushed the probability **higher** (increasing the risk of churn).
    - **Features in <span style='color:blue;'>blue</span>** pushed the probability **lower** (decreasing the risk of churn).
    The size of the bar represents the impact of that feature.
    """, unsafe_allow_html=True)
    
    # Calculate SHAP values for the specific prediction
    shap_values = explainer.shap_values(processed_features)
    
    # --- FIX FOR SHAP PLOT DIMENSION ERROR ---
    # Get the feature names *after* one-hot encoding from the preprocessor
    feature_names_out = preprocessor.get_feature_names_out()
    
    # Create a new SHAP Explanation object with the correct data and feature names
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=processed_features, # Use the processed data
        feature_names=feature_names_out  # Use the correct feature names
    )

    # Create the SHAP force plot using the new explanation object
    # We pass the first explanation (for the first prediction)
    force_plot = shap.force_plot(
    base_value=shap_explanation.base_values,
    shap_values=shap_explanation.values,
    features=shap_explanation.data,
    feature_names=shap_explanation.feature_names
)
    
    # Use the streamlit_shap library to display the plot
    st_shap(force_plot, height=200, width=1000)