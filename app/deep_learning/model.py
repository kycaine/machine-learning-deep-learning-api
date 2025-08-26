import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def build_model():
    model = Sequential([
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)   # output regression
    ])
    
    model.compile(
        optimizer=Adam(0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model
