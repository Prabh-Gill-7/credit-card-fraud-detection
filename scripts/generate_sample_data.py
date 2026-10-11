"""
Synthetic Sample Generator for Credit Card Fraud Detection
==========================================================
Generates a synthetic sample 'creditcard.csv' for immediate testing
matching Kaggle's schema (Time, V1..V28, Amount, Class).

Usage:
    python scripts/generate_sample_data.py
"""
import os
import numpy as np
import pandas as pd

def generate_sample_creditcard_csv(output_path=None, n_samples=5000, fraud_ratio=0.01):
    if output_path is None:
        # Default to data/creditcard.csv relative to project root or current directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, "creditcard.csv")

    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    time_col = np.sort(np.random.uniform(0, 172800, size=n_samples))
    
    # 28 PCA features (V1 through V28)
    v_features_legit = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    v_features_fraud = np.random.normal(loc=-1.5, scale=2.5, size=(n_fraud, 28))
    v_features = np.vstack([v_features_legit, v_features_fraud])

    # Amount column
    amount_legit = np.random.exponential(scale=80, size=n_legit)
    amount_fraud = np.random.exponential(scale=150, size=n_fraud)
    amount = np.concatenate([amount_legit, amount_fraud])

    # Class target
    classes = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.column_stack([time_col, v_features, amount, classes])[indices]

    df = pd.DataFrame(data, columns=columns)
    df["Class"] = df["Class"].astype(int)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic '{output_path}' with {len(df):,} transactions ({df['Class'].sum()} frauds).")

if __name__ == "__main__":
    generate_sample_creditcard_csv()
