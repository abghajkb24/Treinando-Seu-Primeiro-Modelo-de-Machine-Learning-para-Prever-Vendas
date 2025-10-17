#!/usr/bin/env python3
"""
Treina um modelo de regressão para prever 'sales'.
Uso:
python src/train.py --data-path data/train.csv --model random_forest --output-dir models
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocess import load_data, create_features, train_test_split_time, prepare_X_y, fit_scaler, scale_X

def get_model(name: str):
    name = name.lower()
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    if name == "xgboost":
        return XGBRegressor(n_estimators=100, random_state=42, tree_method="hist", verbosity=0)
    raise ValueError(f"Modelo desconhecido: {name}")

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}

def main(data_path, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(data_path)
    df = create_features(df)
    train, test = train_test_split_time(df, test_days=28)
    X_train, y_train = prepare_X_y(train)
    X_test, y_test = prepare_X_y(test)

    scaler, numeric_cols = fit_scaler(X_train)
    X_train_scaled = scale_X(X_train, scaler, numeric_cols)
    X_test_scaled = scale_X(X_test, scaler, numeric_cols)

    model = get_model(model_name)
    print("Treinando modelo:", model_name)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    metrics = evaluate(y_test, preds)
    print("Avaliação no conjunto de teste:", metrics)

    # salvar artefatos
    joblib.dump(model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    # salvar colunas numéricas (para reuso no deploy)
    joblib.dump(numeric_cols, os.path.join(output_dir, "numeric_cols.joblib"))
    print("Modelos salvos em", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model", dest="model_name", default="random_forest", choices=["linear", "random_forest", "xgboost"])
    parser.add_argument("--output-dir", dest="output_dir", default="models")
    args = parser.parse_args()
    main(args.data_path, args.model_name, args.output_dir)