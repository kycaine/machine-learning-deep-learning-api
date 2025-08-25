import numpy as np
from .model import build_model
from sklearn.model_selection import train_test_split

def train_dnn(X: np.ndarray, y: np.ndarray):
    model = build_model(X.shape[1])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    model.save("models/dl/deep_model.h5")
    return model, history.history
