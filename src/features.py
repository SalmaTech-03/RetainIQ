import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the raw Telco churn data.

    Args:
        df: Raw pandas DataFrame.

    Returns:
        A cleaned pandas DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()

    # Convert 'TotalCharges' to numeric, filling missing values.
    # Blanks in TotalCharges appear for new customers (tenure=0),
    # so their total charges are logically 0.
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['TotalCharges'].fillna(0, inplace=True)

    # Convert binary 'Churn' to 0/1
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Drop customerID as it's an identifier, not a feature
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)

    return df_clean

def create_feature_pipeline() -> ColumnTransformer:
    """
    Creates a scikit-learn pipeline for feature engineering.
    This pipeline handles scaling for numeric features and one-hot encoding
    for categorical features.
    
    Returns:
        A scikit-learn ColumnTransformer object.
    """
    # Define numeric and categorical features
    # Note: We exclude 'Churn' from the feature list as it is the target variable.
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any), though we've defined all
    )

    return preprocessor