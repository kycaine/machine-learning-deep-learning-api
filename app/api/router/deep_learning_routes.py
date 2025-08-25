from fastapi import APIRouter, UploadFile, File
import pandas as pd
import numpy as np
from app.deep_learning.trainer import train_dnn
from app.deep_learning.predictor import predict_with_dnn

router = APIRouter()

@router.post("/dl/train", tags=["Deep Learning"])
def train_dl(data: dict):
    df = pd.read_csv("data/uploaded.csv") 
    X = df[data["feature_columns"]].values
    y = df[data["target_column"]].values
    model, history = train_dnn(X, y)
    return {"status": "training complete", "history": history}

@router.post("/dl/predict", tags=["Deep Learning"])
def predict_dl(data: dict):
    df = pd.read_csv("data/uploaded.csv")
    X = df[data["feature_columns"]].values
    preds = predict_with_dnn(X)
    return {"predictions": preds.tolist()}
