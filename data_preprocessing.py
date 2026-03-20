import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load and clean the initial dataset"""
    data = pd.read_csv(file_path)
    df = pd.DataFrame(data)
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print("Original Data:")
    print(df)
    print("\nNumber of rows in data:", len(data))
    print("\n")
    return df

def preprocess_data(df):
    """Preprocess the data by encoding categorical variables and scaling"""
    df_processed = df.copy()
    label_encoders = {}
    scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
    imputer = SimpleImputer(strategy='median')  # Changed to median for better robustness

    # Encode categorical variables
    categorical_columns = ['gender', 'occupation', 'field', 'income_level', 'current_phone_brand']

    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le

    # Handle missing values
    df_processed = pd.DataFrame(imputer.fit_transform(df_processed), columns=df_processed.columns)

    # Scale all features
    scaled_data = scaler.fit_transform(df_processed)

    return scaled_data, df_processed.columns.tolist()