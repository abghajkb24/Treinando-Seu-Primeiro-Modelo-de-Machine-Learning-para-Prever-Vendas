"""
Funções de pré-processamento e engenharia de features simples para previsão de vendas.
- cria colunas de data (year, month, day, weekday)
- cria lags e médias móveis por store/product
- retorna X, y e a pipeline (scaler)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

LAG_DAYS = [1, 7, 14]
ROLL_WINDOWS = [7, 14]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    # group features (lag and rolling) per store/product
    df_grouped = []
    for (s, p), g in df.groupby(["store_id", "product_id"]):
        g = g.sort_values("date").copy()
        for l in LAG_DAYS:
            g[f"lag_{l}"] = g["sales"].shift(l)
        for w in ROLL_WINDOWS:
            g[f"roll_mean_{w}"] = g["sales"].shift(1).rolling(window=w, min_periods=1).mean()
        g["price_diff"] = g["price"] - g["price"].shift(1)
        g = g.fillna(0)
        df_grouped.append(g)
    df2 = pd.concat(df_grouped, axis=0).sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return df2

def train_test_split_time(df: pd.DataFrame, test_days: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep last test_days per store/product as test
    train_idx = []
    test_idx = []
    for (s, p), g in df.groupby(["store_id", "product_id"]):
        g = g.sort_values("date")
        if len(g) <= test_days:
            train_idx += g.index.tolist()
        else:
            test_idx += g.index[-test_days:].tolist()
            train_idx += g.index[:-test_days].tolist()
    train = df.loc[train_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)
    return train, test

def prepare_X_y(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        # default features
        feature_cols = [
            "store_id", "product_id", "price", "promo",
            "year", "month", "day", "weekday",
        ] + [f"lag_{l}" for l in LAG_DAYS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["price_diff"]
    X = df[feature_cols].copy()
    y = df["sales"].copy()
    return X, y

def fit_scaler(X_train: pd.DataFrame):
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler.fit(X_train[numeric_cols])
    return scaler, numeric_cols

def scale_X(X: pd.DataFrame, scaler: StandardScaler, numeric_cols):
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.transform(X_scaled[numeric_cols])
    return X_scaled