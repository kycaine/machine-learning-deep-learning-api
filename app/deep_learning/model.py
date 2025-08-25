from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1) 
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model
