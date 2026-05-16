"""
03_preprocess.py
-----------------
Preprocessing pipeline:
  - Feature scaling (StandardScaler on Amount & Time)
  - Train/test split (stratified)
  - Three sampling strategies: None, SMOTE, RandomUnderSampler
  - Saves processed splits to data/processed/

Run after: 01_download_data.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

DATA_DIR      = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2


def load_data():
    csv = DATA_DIR / "creditcard.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Run 01_download_data.py first.")
    df = pd.read_csv(csv)
    print(f"[OK] Loaded {len(df):,} rows")
    return df


def scale_features(df):
    """Scale Amount and Time; V1-V28 are already PCA-transformed."""
    df = df.copy()
    scaler_amount = StandardScaler()
    scaler_time   = StandardScaler()
    df["Amount_scaled"] = scaler_amount.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler_time.fit_transform(df[["Time"]])
    df.drop(["Amount", "Time"], axis=1, inplace=True)

    # Save scalers
    with open(PROCESSED_DIR / "scaler_amount.pkl", "wb") as f:
        pickle.dump(scaler_amount, f)
    with open(PROCESSED_DIR / "scaler_time.pkl", "wb") as f:
        pickle.dump(scaler_time, f)

    print("[OK] Features scaled. Scalers saved.")
    return df


def split_data(df):
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[OK] Split: train={len(X_train):,} | test={len(X_test):,}")
    print(f"     Train fraud: {y_train.sum()} ({y_train.mean():.4%})")
    print(f"     Test fraud : {y_test.sum()} ({y_test.mean():.4%})")
    return X_train, X_test, y_train, y_test


def apply_sampling(X_train, y_train):
    """Return dict of (X, y) for different sampling strategies."""
    results = {}

    # 1. No sampling (imbalanced)
    results["no_sampling"] = (X_train.copy(), y_train.copy())
    print(f"[OK] No sampling — fraud: {y_train.sum():,} / {len(y_train):,}")

    # 2. SMOTE oversampling
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    results["smote"] = (X_sm, y_sm)
    print(f"[OK] SMOTE — fraud: {y_sm.sum():,} / {len(y_sm):,}")

    # 3. Random undersampling
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_ru, y_ru = rus.fit_resample(X_train, y_train)
    results["undersample"] = (X_ru, y_ru)
    print(f"[OK] Undersample — fraud: {y_ru.sum():,} / {len(y_ru):,}")

    return results


def save_splits(X_test, y_test, sampling_results):
    # Save test set (common across all experiments)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    # Save each sampling variant
    for name, (X, y) in sampling_results.items():
        pd.DataFrame(X).to_csv(PROCESSED_DIR / f"X_train_{name}.csv", index=False)
        pd.Series(y).to_csv(PROCESSED_DIR / f"y_train_{name}.csv", index=False)

    print(f"[OK] All splits saved to {PROCESSED_DIR}")


def main():
    df = load_data()
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    sampling_results = apply_sampling(X_train, y_train)
    save_splits(X_test, y_test, sampling_results)
    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    main()
