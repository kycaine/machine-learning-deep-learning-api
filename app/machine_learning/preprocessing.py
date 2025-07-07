import os
from datetime import datetime

import dateparser
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.config.constants import *


def clean_data(file_path: str, columns_schema: dict, original_filename: str):
    df = pd.read_csv(file_path)
    
    requested_columns = list(columns_schema.keys())
    
    #validation column
    missing = [col for col in requested_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{MISSING_UNMATCH_COLLUMN}: {missing}")
    
    df = df[requested_columns]

    for col, dtype in columns_schema.items():
        try:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "str":
                df[col] = df[col].astype(str)
        except Exception as e:
            print(f"Error converting column {col} to {dtype}: {e}")

    # remove negative values
    for col, dtype in columns_schema.items():
        if dtype in ["int", "float"]:
            df.loc[df[col] < 0, col] = np.nan

    # clean dates
    date_cols = [col for col, dtype in columns_schema.items() if "date" in col.lower()]
    for date_col in date_cols:
        try:
            df[date_col] = df[date_col].apply(lambda x: dateparser.parse(str(x)) if pd.notnull(x) else None)
            df[date_col] = df[date_col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
        except Exception as e:
            print(f"Error parsing date column {date_col}: {e}")

    df_cleaned = df.dropna().drop_duplicates()
    
    name_only, _ = os.path.splitext(original_filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_filename = f"cleaned_{name_only}_{timestamp}.csv"
    cleaned_path = os.path.join(OUTPUTS_CLEANED, cleaned_filename)
    df_cleaned.to_csv(cleaned_path, index=False)

    summary = {
        "kept_columns": requested_columns,
        "converted_types": columns_schema,
        "original_rows": len(df),
        "rows_after_clean": len(df_cleaned),
        "num_rows_dropped": len(df) - len(df_cleaned)
    }

    return cleaned_path, summary

def feature_engineering(df: pd.DataFrame, filename: str, feature_columns: list, columns_schema: dict):
    processed_df = preprocess_data(df, feature_columns, columns_schema)
    processed_dir = OUTPUTS_FEATURE_ENGINNERING

    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(processed_dir, processed_filename)
    processed_df.to_csv(processed_path, index=False)

    return processed_filename, processed_df

def preprocess_data(df: pd.DataFrame, feature_columns: list, columns_schema: dict):
    df = df[feature_columns].copy()

    date_columns = [col for col in feature_columns if columns_schema.get(col) in ['str', 'datetime'] and 'date' in col.lower()]
    for date_col in date_columns:
        df = generate_date_features(df, date_col)

    categorical_columns = [col for col in df.columns if columns_schema.get(col) == 'str' and col not in date_columns]
    df = encode_categorical_columns(df, categorical_columns)

    numerical_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    df = scale_numerical_columns(df, numerical_columns)

    return df

def generate_date_features(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df[f"{date_column}_dayofweek"] = df[date_column].dt.dayofweek
    df[f"{date_column}_month"] = df[date_column].dt.month
    df[f"{date_column}_day"] = df[date_column].dt.day
    df.drop(columns=[date_column], inplace=True)
    return df

# do encode if it category data
def encode_categorical_columns(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)

def scale_numerical_columns(df, numerical_columns):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df