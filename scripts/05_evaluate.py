"""
05_evaluate.py
---------------
Generates evaluation visualizations:
  - ROC curves for all models
  - Precision-Recall curves
  - Confusion matrices
  - Metric comparison bar charts
  - Feature importance (for tree models)
  - Task decomposition analysis chart

Run after: 04_train_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, average_precision_score
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "outputs" / "models"
PLOTS_DIR     = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "logistic_regression": "#185FA5",
    "decision_tree":       "#888780",
    "random_forest":       "#0F6E56",
    "xgboost":             "#D85A30",
}
SAMPLING_LINESTYLE = {
    "no_sampling": "-",
    "smote":       "--",
    "undersample": ":",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree":       "Decision Tree",
    "random_forest":       "Random Forest",
    "xgboost":             "XGBoost",
}
SAMPLING_LABELS = {
    "no_sampling": "No Sampling",
    "smote":       "SMOTE",
    "undersample": "Undersample",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def load_test():
    X = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X, y


def load_model(model_name, strategy):
    path = MODELS_DIR / f"{model_name}_{strategy}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metrics():
    return pd.read_csv(MODELS_DIR / "all_metrics.csv")


# ── ROC Curves ────────────────────────────────────────────────────────────
def plot_roc_curves(X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ROC Curves by Sampling Strategy", fontsize=13, fontweight="bold")

    strategies = ["no_sampling", "smote", "undersample"]
    models     = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

    for ax, strategy in zip(axes, strategies):
        for model_name in models:
            try:
                model  = load_model(model_name, strategy)
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=MODEL_COLORS[model_name], linewidth=1.8,
                        label=f"{MODEL_LABELS[model_name]} ({roc_auc:.3f})")
            except Exception:
                pass

        ax.plot([0,1],[0,1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(SAMPLING_LABELS[strategy])
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 07_roc_curves.png")


# ── Precision-Recall Curves ───────────────────────────────────────────────
def plot_pr_curves(X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Precision-Recall Curves by Sampling Strategy", fontsize=13, fontweight="bold")

    strategies = ["no_sampling", "smote", "undersample"]
    models     = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]

    for ax, strategy in zip(axes, strategies):
        baseline = y_test.mean()
        ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=0.8,
                   label=f"Baseline ({baseline:.3f})")
        for model_name in models:
            try:
                model  = load_model(model_name, strategy)
                y_prob = model.predict_proba(X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                ap = average_precision_score(y_test, y_prob)
                ax.plot(rec, prec, color=MODEL_COLORS[model_name], linewidth=1.8,
                        label=f"{MODEL_LABELS[model_name]} (AP={ap:.3f})")
            except Exception:
                pass

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(SAMPLING_LABELS[strategy])
        ax.legend(fontsize=8)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_pr_curves.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 08_pr_curves.png")


# ── Confusion Matrices ────────────────────────────────────────────────────
def plot_confusion_matrices(X_test, y_test):
    models     = ["logistic_regression", "random_forest", "xgboost"]
    strategies = ["no_sampling", "smote", "undersample"]

    fig, axes = plt.subplots(len(models), len(strategies), figsize=(14, 10))
    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")

    for row, model_name in enumerate(models):
        for col, strategy in enumerate(strategies):
            ax = axes[row][col]
            try:
                model  = load_model(model_name, strategy)
                y_pred = model.predict(X_test)
                cm     = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Pred Legit","Pred Fraud"],
                            yticklabels=["True Legit","True Fraud"],
                            ax=ax, cbar=False, linewidths=0.5,
                            annot_kws={"size": 10})
                ax.set_title(f"{MODEL_LABELS[model_name]}\n{SAMPLING_LABELS[strategy]}",
                             fontsize=9)
            except Exception as e:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_confusion_matrices.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 09_confusion_matrices.png")


# ── Metric Comparison ─────────────────────────────────────────────────────
def plot_metric_comparison(metrics_df):
    metrics_to_plot = ["roc_auc", "avg_precision", "f1_fraud", "recall_fraud", "precision_fraud"]
    metric_labels   = ["ROC-AUC", "Avg Precision", "F1 (Fraud)", "Recall (Fraud)", "Precision (Fraud)"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))
    fig.suptitle("Model Performance Comparison Across Metrics", fontsize=13, fontweight="bold")

    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        pivot = metrics_df.pivot_table(index="model", columns="sampling", values=metric)
        pivot.index = [MODEL_LABELS.get(i, i) for i in pivot.index]
        x = np.arange(len(pivot))
        width = 0.25
        colors = ["#185FA5", "#0F6E56", "#D85A30"]

        for i, (col, color) in enumerate(zip(["no_sampling","smote","undersample"], colors)):
            if col in pivot.columns:
                bars = ax.bar(x + i*width, pivot[col], width, label=SAMPLING_LABELS[col],
                              color=color, alpha=0.85, edgecolor="white")

        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1.1)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_metric_comparison.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 10_metric_comparison.png")


# ── Feature Importance ────────────────────────────────────────────────────
def plot_feature_importance(X_test):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Feature Importance (Best Models)", fontsize=13, fontweight="bold")

    for ax, (model_name, strategy) in enumerate([("random_forest","smote"), ("xgboost","smote")]):
        axh = axes[ax]
        try:
            model = load_model(model_name, strategy)
            importances = model.feature_importances_
            feat_names  = X_test.columns.tolist()
            df_fi = pd.DataFrame({"feature": feat_names, "importance": importances})
            df_fi = df_fi.sort_values("importance", ascending=True).tail(15)
            color = MODEL_COLORS[model_name]
            axh.barh(df_fi["feature"], df_fi["importance"], color=color, alpha=0.85, edgecolor="none")
            axh.set_xlabel("Importance score")
            axh.set_title(f"{MODEL_LABELS[model_name]} + SMOTE", fontsize=11)
        except Exception as e:
            axh.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 11_feature_importance.png")


# ── Task Decomposition Analysis Chart ────────────────────────────────────
def plot_task_decomposition_analysis(metrics_df):
    """
    Maps the research question directly:
    Baseline = Logistic Regression no_sampling
    Chain-of-Thought = Random Forest no_sampling
    Task Decomposed = XGBoost + SMOTE
    """
    agents = {
        "Baseline Agent\n(Logistic Reg, No Sampling)":    ("logistic_regression", "no_sampling"),
        "Intermediate Agent\n(Random Forest, No Sampling)": ("random_forest",        "no_sampling"),
        "Task Decomposed Agent\n(XGBoost + SMOTE)":         ("xgboost",              "smote"),
    }

    metric_cols  = ["roc_auc", "avg_precision", "f1_fraud", "recall_fraud", "precision_fraud"]
    metric_names = ["ROC-AUC", "Avg Precision", "F1-Fraud", "Recall", "Precision"]

    rows = []
    for label, (m, s) in agents.items():
        row = metrics_df[(metrics_df.model==m) & (metrics_df.sampling==s)]
        if not row.empty:
            vals = [row[c].values[0] for c in metric_cols]
            rows.append({"Agent": label, **dict(zip(metric_names, vals))})

    df_plot = pd.DataFrame(rows).set_index("Agent")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#888780", "#185FA5", "#0F6E56"]

    for i, (agent, color) in enumerate(zip(df_plot.index, colors)):
        vals = df_plot.loc[agent].values
        bars = ax.bar(x + i*width, vals, width, label=agent, color=color, alpha=0.88, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Effect of Task Decomposition on Agent Performance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_task_decomposition_effect.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 12_task_decomposition_effect.png")


def main():
    print("[START] Generating evaluation plots...")
    X_test, y_test = load_test()
    metrics_df     = load_metrics()

    plot_roc_curves(X_test, y_test)
    plot_pr_curves(X_test, y_test)
    plot_confusion_matrices(X_test, y_test)
    plot_metric_comparison(metrics_df)
    plot_feature_importance(X_test)
    plot_task_decomposition_analysis(metrics_df)

    print(f"\n[DONE] All evaluation plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
