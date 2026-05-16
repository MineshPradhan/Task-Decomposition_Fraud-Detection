"""
04_train_models.py
-------------------
Trains four models across three sampling strategies:
  Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
  Sampling: no_sampling, smote, undersample

This directly maps to the research question:
  - Baseline agent  → Logistic Regression (no_sampling)
  - Intermediate    → Decision Tree / Random Forest
  - Best performer  → XGBoost + SMOTE (task-decomposed equivalent)

Saves all trained models + metrics to outputs/models/
Run after: 03_preprocess.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "outputs" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced", C=0.1
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=8, random_state=42, class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        n_jobs=-1, class_weight="balanced"
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=577,  # approx class imbalance ratio
        random_state=42, eval_metric="aucpr",
        verbosity=0, use_label_encoder=False
    ),
}

SAMPLING_STRATEGIES = ["no_sampling", "smote", "undersample"]


def load_split(strategy):
    X_train = pd.read_csv(PROCESSED_DIR / f"X_train_{strategy}.csv")
    y_train = pd.read_csv(PROCESSED_DIR / f"y_train_{strategy}.csv").squeeze()
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test


def evaluate(model, X_test, y_test, model_name, strategy):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":             model_name,
        "sampling":          strategy,
        "roc_auc":           round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision":     round(average_precision_score(y_test, y_proba), 4),
        "f1_fraud":          round(f1_score(y_test, y_pred, pos_label=1), 4),
        "precision_fraud":   round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall_fraud":      round(recall_score(y_test, y_pred, pos_label=1), 4),
        "f1_macro":          round(f1_score(y_test, y_pred, average="macro"), 4),
        "cm":                confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def train_all():
    all_metrics = []

    for strategy in SAMPLING_STRATEGIES:
        print(f"\n{'='*55}")
        print(f"  Sampling strategy: {strategy.upper()}")
        print(f"{'='*55}")

        X_train, y_train, X_test, y_test = load_split(strategy)
        print(f"  Train size: {len(X_train):,} | Fraud train: {int(y_train.sum()):,}")

        for model_name, model in MODELS.items():
            print(f"\n  [{model_name}] Training...", end=" ")
            t0 = time.time()
            model.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 2)
            print(f"done in {elapsed}s")

            metrics = evaluate(model, X_test, y_test, model_name, strategy)
            metrics["train_time_s"] = elapsed
            all_metrics.append(metrics)

            print(f"    ROC-AUC: {metrics['roc_auc']:.4f} | "
                  f"Avg-Precision: {metrics['avg_precision']:.4f} | "
                  f"F1-Fraud: {metrics['f1_fraud']:.4f} | "
                  f"Recall: {metrics['recall_fraud']:.4f}")

            # Save model
            model_file = MODELS_DIR / f"{model_name}_{strategy}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)

    # Save all metrics
    metrics_df = pd.DataFrame([{k: v for k, v in m.items() if k != "cm"} for m in all_metrics])
    metrics_df.to_csv(MODELS_DIR / "all_metrics.csv", index=False)

    with open(MODELS_DIR / "all_metrics_full.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics, metrics_df


def main():
    print("\n[START] Training all models...\n")
    all_metrics, metrics_df = train_all()

    print("\n" + "="*55)
    print("  TRAINING COMPLETE — TOP RESULTS BY ROC-AUC")
    print("="*55)
    top = metrics_df.sort_values("roc_auc", ascending=False).head(5)
    print(top[["model", "sampling", "roc_auc", "avg_precision", "f1_fraud", "recall_fraud"]].to_string(index=False))
    print(f"\n[DONE] Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
