#!/usr/bin/env python3
"""
Gera um CSV sintético de vendas para demonstração.

Colunas:
- date (YYYY-MM-DD)
- store_id (int)
- product_id (int)
- price (float)
- promo (0/1)
- sales (float)  <- target

Uso:
python data/generate_synthetic_data.py --out data/train.csv --n-stores 5 --n-products 20 --periods 730
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate(out_path, n_stores=3, n_products=10, periods=365, start_date="2020-01-01", seed=42):
    np.random.seed(seed)
    start = datetime.fromisoformat(start_date)
    rows = []
    for s in range(1, n_stores + 1):
        for p in range(1, n_products + 1):
            # base demand and seasonality per product/store
            base = np.random.uniform(20, 200)
            trend = np.random.uniform(0.0, 0.05)  # slight trend
            season_amp = np.random.uniform(0.1, 0.6)
            for t in range(periods):
                date = start + timedelta(days=t)
                dow = date.weekday()
                # weekly seasonality (weekend effect)
                weekly = 1.2 if dow >= 5 else 1.0
                # yearly seasonality
                day_of_year = date.timetuple().tm_yday
                yearly = 1 + season_amp * np.sin(2 * np.pi * day_of_year / 365)
                price = np.round(np.random.uniform(5, 50), 2)
                promo = np.random.binomial(1, 0.1)  # 10% chance of promo
                noise = np.random.normal(0, base * 0.1)
                sales = base * (1 + trend * t) * weekly * yearly * (0.8 if promo else 1.0) + noise
                sales = max(0, round(sales, 2))
                rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_id": s,
                    "product_id": p,
                    "price": price,
                    "promo": int(promo),
                    "sales": sales
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Gerado {len(df)} linhas em {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", dest="out", required=True, help="Caminho do CSV de saída")
    parser.add_argument("--n-stores", type=int, default=3)
    parser.add_argument("--n-products", type=int, default=10)
    parser.add_argument("--periods", type=int, default=365)
    parser.add_argument("--start-date", dest="start_date", default="2020-01-01")
    args = parser.parse_args()
    generate(args.out, args.n_stores, args.n_products, args.periods, args.start_date)