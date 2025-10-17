"""
FastAPI app para previsão em tempo real.
POST /predict
Recebe JSON com array de registros com colunas: date, store_id, product_id, price, promo
Retorna previsões.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from src.preprocess import create_features, prepare_X_y, scale_X

MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
NUMERIC_COLS_PATH = "models/numeric_cols.joblib"

app = FastAPI(title="Sales Forecast API")

class Record(BaseModel):
    date: str
    store_id: int
    product_id: int
    price: float
    promo: int

class PredictRequest(BaseModel):
    records: List[Record]

@app.on_event("startup")
def load_artifacts():
    global model, scaler, numeric_cols
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        numeric_cols = joblib.load(NUMERIC_COLS_PATH)
    except Exception as e:
        # manter app disponível mesmo sem modelo (útil para desenvolvimento)
        model = None
        scaler = None
        numeric_cols = None
        print("Aviso: artefatos não encontrados:", e)

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado no servidor.")
    df = pd.DataFrame([r.dict() for r in req.records])
    df["date"] = pd.to_datetime(df["date"])
    df_feat = create_features(df)
    X, _ = prepare_X_y(df_feat)
    X_scaled = scale_X(X, scaler, numeric_cols)
    preds = model.predict(X_scaled)
    return {"predictions": preds.tolist()}