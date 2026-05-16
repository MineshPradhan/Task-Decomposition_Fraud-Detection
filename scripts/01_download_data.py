"""
01_download_data.py
--------------------
Downloads the Credit Card Fraud Detection dataset from Kaggle.
If Kaggle credentials are not available, generates a realistic
synthetic dataset with similar statistical properties.

Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = DATA_DIR / "creditcard.csv"


def download_from_kaggle():
    """Attempt to download via kaggle API."""
    try:
        import kaggle
        print("[INFO] Kaggle API found. Downloading dataset...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud",
            path=str(DATA_DIR),
            unzip=True
        )
        print(f"[OK] Dataset saved to {OUTPUT_CSV}")
        return True
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        return False


def generate_synthetic_dataset(n_samples=284807, fraud_ratio=0.00173):
    """
    Generate a synthetic dataset matching the real creditcard.csv schema:
    - 28 PCA-transformed features (V1–V28)
    - Time (seconds from first transaction)
    - Amount (transaction amount)
    - Class (0 = legit, 1 = fraud)
    """
    print(f"[INFO] Generating synthetic dataset ({n_samples:,} rows)...")
    np.random.seed(42)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions ---
    legit_V = np.random.randn(n_legit, 28)
    # Introduce realistic correlations
    for i in range(1, 28):
        legit_V[:, i] += 0.05 * legit_V[:, i - 1]

    legit_amount = np.random.exponential(scale=88, size=n_legit).clip(0.01, 25691.16)
    legit_time = np.sort(np.random.uniform(0, 172792, n_legit))

    # --- Fraudulent transactions ---
    fraud_V = np.random.randn(n_fraud, 28) * 1.5
    # Fraud has distinct patterns in certain components
    fraud_V[:, 0] += -3.5   # V1 shift
    fraud_V[:, 1] += 2.8    # V2 shift
    fraud_V[:, 2] += -2.2   # V3 shift
    fraud_V[:, 3] += 1.5    # V4 shift
    fraud_V[:, 13] += -2.0  # V14 shift (key fraud indicator)
    fraud_V[:, 16] += -1.8  # V17 shift

    fraud_amount = np.random.exponential(scale=120, size=n_fraud).clip(0.01, 2125.87)
    fraud_time = np.random.uniform(0, 172792, n_fraud)

    # --- Combine ---
    V_all = np.vstack([legit_V, fraud_V])
    amounts = np.concatenate([legit_amount, fraud_amount])
    times = np.concatenate([legit_time, fraud_time])
    labels = np.array([0] * n_legit + [1] * n_fraud)

    cols = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(V_all, columns=cols)
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"] = labels

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Synthetic dataset saved to {OUTPUT_CSV}")
    print(f"     Total: {len(df):,} | Fraud: {labels.sum():,} | Ratio: {labels.mean():.4%}")
    return df


def main():
    if OUTPUT_CSV.exists():
        print(f"[INFO] Dataset already exists at {OUTPUT_CSV}. Skipping download.")
        df = pd.read_csv(OUTPUT_CSV)
        print(f"       Shape: {df.shape} | Fraud: {df['Class'].sum():,}")
        return

    # Try Kaggle first
    success = download_from_kaggle()

    if not success:
        print("[INFO] Falling back to synthetic data generation...")
        generate_synthetic_dataset()


if __name__ == "__main__":
    main()
