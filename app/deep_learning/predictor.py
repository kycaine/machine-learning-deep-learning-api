from tensorflow.keras.models import load_model
import numpy as np

def predict_with_dnn(X: np.ndarray):
    model = load_model("models/dl/deep_model.h5")
    predictions = model.predict(X)
    return predictions
