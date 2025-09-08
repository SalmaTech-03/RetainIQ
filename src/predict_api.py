from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# Define the input data model using Pydantic for validation
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., alias='SeniorCitizen', ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Initialize the FastAPI app
app = FastAPI(title="ChurnBuster API", version="1.0")

# --- Load the trained model and preprocessor on startup ---

# Get the absolute path to the directory where this script is located
# e.g., /app/src
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the models directory
# e.g., /app/src -> /app -> /app/models
MODEL_DIR = os.path.join(SCRIPT_DIR, '..', 'models')

# Construct the full paths to the model and preprocessor files
model_path = os.path.join(MODEL_DIR, 'xgb_model_v1.0.joblib')
preprocessor_path = os.path.join(MODEL_DIR, 'preprocessor_v1.0.joblib')

# Load the artifacts
try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print("Model and preprocessor loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or preprocessor not found at {MODEL_DIR}")
    # In a real app, you might want to exit or handle this more gracefully
    model = None
    preprocessor = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the ChurnBuster Prediction API"}

@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """
    Accepts customer features and returns a churn prediction.
    """
    try:
        # Convert input data to a pandas DataFrame
        # The `alias` in Pydantic handles the capital letter in 'SeniorCitizen'
        input_df = pd.DataFrame([features.dict(by_alias=True)])
        
        # Preprocess the data using the loaded pipeline
        processed_features = preprocessor.transform(input_df)
        
        # Make a prediction (returns the probability of the positive class '1')
        probability = model.predict_proba(processed_features)[0, 1]
        
        # Determine risk band based on probability
        if probability > 0.5:
            risk_band = "High"
        elif probability > 0.25:
            risk_band = "Medium"
        else:
            risk_band = "Low"

        return {
            "churn_probability": float(probability),
            "risk_band": risk_band
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run the API:
# 1. cd into the `src` directory
# 2. Run: uvicorn predict_api:app --reload