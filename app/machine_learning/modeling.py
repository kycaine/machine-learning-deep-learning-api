# app/machine_learning/modeling.py
import os
import pickle
from datetime import datetime
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.config.constants import OUTPUTS_MODELS  
def train_model(df, target_column: str, feature_columns: list, base_filename: str = "model"):
    X = df[feature_columns]
    y = df[target_column]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"trained_{base_filename}_{timestamp}.pkl"
    model_path = os.path.join(OUTPUTS_MODELS, model_filename).replace("\\", "/")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"IM HERE : {model_path}")

    summary = {
        "mae": mae,
        "rmse": rmse
    }

    return model_path, summary