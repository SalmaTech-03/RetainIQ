import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, classification_report
import joblib
import os

# MLFLOW: Import the mlflow library
import mlflow
import mlflow.xgboost

# Import the functions from your feature engineering script
from features import preprocess_data, create_feature_pipeline

# MLFLOW: Set a name for our experiment
mlflow.set_experiment("ChurnBuster XGBoost")

print("--- Starting Model Training Pipeline ---")

# --- 1. Load Data ---
print("Step 1: Loading data...")
try:
    df = pd.read_csv('C:\\Users\\syedm\\OneDrive\\ChurnBuster — Predictive Churn Scoring + Targeted Retention Engine\\churnbuster\\data\\raw\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: The dataset file was not found...")
    exit()

# --- 2. Preprocess Data ---
print("Step 2: Preprocessing data...")
df_clean = preprocess_data(df)
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

# --- 3. Split Data ---
print("Step 3: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Feature Engineering ---
print("Step 4: Applying feature engineering pipeline...")
preprocessor = create_feature_pipeline()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Feature engineering complete.")

# MLFLOW: Start an MLflow run. Everything inside this block will be logged.
with mlflow.start_run():
    print("Step 5: Training Production Model (XGBoost)...")
    
    # Define model parameters so we can log them
    params = {
        'objective': 'binary:logistic',
        'scale_pos_weight': y_train.value_counts()[0] / y_train.value_counts()[1],
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_estimators': 150, # Example parameter
        'max_depth': 4       # Example parameter
    }
    
    # MLFLOW: Log the parameters
    mlflow.log_params(params)
    
    xgb_model = XGBClassifier(**params)
    xgb_model.fit(X_train_processed, y_train)
    print("Model training complete.")

    # --- 6. Model Evaluation ---
    print("Step 6: Evaluating model...")
    y_pred_proba = xgb_model.predict_proba(X_test_processed)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # MLFLOW: Log the metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("brier_score", brier)
    
    print(f"\n--- XGBoost Metrics ---")
    print(f"ROC AUC:     {roc_auc:.4f}")
    print(f"PR AUC:      {pr_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")

    # MLFLOW: Log the trained model and preprocessor artifacts
    print("MLFLOW: Logging model artifacts...")
    mlflow.xgboost.log_model(xgb_model, "xgb_model")
    joblib.dump(preprocessor, "preprocessor.joblib")
    mlflow.log_artifact("preprocessor.joblib")

# --- 7. Save Final Artifacts for API ---
print("\nStep 7: Saving model and preprocessor for API deployment...")
os.makedirs('../models', exist_ok=True)
joblib.dump(xgb_model, '../models/xgb_model_v1.0.joblib')
joblib.dump(preprocessor, '../models/preprocessor_v1.0.joblib')
print("Artifacts saved to the 'models/' directory.")
print("\n--- Model Training Pipeline Finished ---")