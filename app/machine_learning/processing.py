import os
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from zipfile import ZipFile

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fastapi import UploadFile
from typing import Dict

from app.config.constants import *


def train_and_predict(data_file: UploadFile, metadata: Dict):
    target_column = metadata["target_column"]
    feature_columns = metadata["feature_columns"]
    column_types = metadata["columns"]

    df = pd.read_csv(data_file.file)

    for col, dtype in column_types.items():
        if col in df.columns:
            if dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif dtype == "str":
                df[col] = df[col].astype(str)

    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    evaluation = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    pred_df = pd.DataFrame({
        "Actual": y_test.reset_index(drop=True),
        "Predicted": y_pred
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{timestamp}.csv"
    output_path = f"{OUTPUTS_PREDICT}/{output_filename}"
    os.makedirs(OUTPUTS_PREDICT, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    
    #create visualization
    os.makedirs(OUTPUTS_VISUAL, exist_ok=True)
    vis_files = []

    #visual 1 actual and predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pred_df["Actual"], y=pred_df["Predicted"])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plot1 = os.path.join(OUTPUTS_VISUAL, f"actual_vs_predicted_{timestamp}.png")
    plt.savefig(plot1)
    vis_files.append(plot1)
    plt.close()

    #visual 2 residuals distribution
    residuals = pred_df["Actual"] - pred_df["Predicted"]
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title("Residuals Distribution")
    plot2 = os.path.join(OUTPUTS_VISUAL, f"residuals_{timestamp}.png")
    plt.savefig(plot2)
    vis_files.append(plot2)
    plt.close()

    #visual 3 feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance")
    plot3 = os.path.join(OUTPUTS_VISUAL, f"feature_importance_{timestamp}.png")
    plt.savefig(plot3)
    vis_files.append(plot3)
    plt.close()

    #zip visual
    zip_filename = f"visual_predictions_{timestamp}.zip"
    zip_path = os.path.join(OUTPUTS_VISUAL, zip_filename).replace("\\", "/")
    with ZipFile(zip_path, 'w') as zipf:
        for file in vis_files:
            zipf.write(file, os.path.basename(file))

    return evaluation, y_pred.tolist(), output_path, zip_path
