import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss
from features import preprocess_data

def evaluate_model(model_path: str, preprocessor_path: str, test_data_path: str):
    """Loads a model and evaluates it on test data."""
    print("--- Starting Model Evaluation ---")
    
    # Load model and preprocessor
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Load and preprocess test data
    df_test = pd.read_csv(test_data_path)
    df_test_clean = preprocess_data(df_test)
    
    X_test = df_test_clean.drop('Churn', axis=1)
    y_test = df_test_clean['Churn']
    
    # Apply the same transformations
    X_test_processed = preprocessor.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    print(f"\n--- Performance Metrics ---")
    print(f"ROC AUC:     {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("--- Evaluation Complete ---")

if __name__ == '__main__':
    # Example usage:
    evaluate_model(
        model_path='../models/xgb_model_v1.0.joblib',
        preprocessor_path='../models/preprocessor_v1.0.joblib',
        # Assuming you have a separate test set, but we'll reuse the full data for this example
        test_data_path='../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )