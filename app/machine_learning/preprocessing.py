import os
from datetime import datetime

import dateparser
import numpy as np
import pandas as pd

from app.config.paths import OUTPUTS_CLEANED


def clean_data_strict(file_path: str, columns_schema: dict, original_filename: str):
    df = pd.read_csv(file_path)
    
    requested_columns = list(columns_schema.keys())
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
