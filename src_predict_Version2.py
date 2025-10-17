"""
Utilitário simples para carregar o modelo salvo e predizer em novos registros.
Exemplo de uso:
python src/predict.py --model-path models/model.joblib --scaler-path models/scaler.joblib --input csv_de_entrada.csv
"""
import argparse
import pandas as pd
import joblib
from src.preprocess import create_features, prepare_X_y, scale_X

def predict(model_path, scaler_path, numeric_cols_path, input_path, output_path=None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    numeric_cols = joblib.load(numeric_cols_path)

    df = pd.read_csv(input_path, parse_dates=["date"])
    df_feat = create_features(df)
    X, _ = prepare_X_y(df_feat)
    # garantir colunas numéricas salvas
    X_scaled = scale_X(X, scaler, numeric_cols)
    preds = model.predict(X_scaled)
    df_out = df.copy()
    df_out["prediction"] = preds
    if output_path:
        df_out.to_csv(output_path, index=False)
        print(f"Predições salvas em {output_path}")
    else:
        print(df_out.head())
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scaler-path", required=True)
    parser.add_argument("--numeric-cols-path", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=False)
    args = parser.parse_args()
    predict(args.model_path, args.scaler_path, args.numeric_cols_path, args.input, args.output)