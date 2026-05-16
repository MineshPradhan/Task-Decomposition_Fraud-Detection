"""
02_eda.py
----------
Exploratory Data Analysis for the Credit Card Fraud Detection dataset.
Produces plots saved to outputs/plots/ and prints key statistics.

Run after: 01_download_data.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

DATA_DIR  = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"legit": "#185FA5", "fraud": "#D85A30"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def load_data():
    csv = DATA_DIR / "creditcard.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Dataset not found at {csv}. Run 01_download_data.py first.")
    df = pd.read_csv(csv)
    print(f"[OK] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ── Plot 1: Class imbalance ────────────────────────────────────────────────
def plot_class_distribution(df):
    counts = df["Class"].value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = [PALETTE["legit"], PALETTE["fraud"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class Distribution — Transaction Dataset", fontsize=13, fontweight="bold", y=1.01)

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Absolute counts")
    axes[0].set_ylim(0, counts.max() * 1.15)

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct="%1.3f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class proportions")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_class_distribution.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 01_class_distribution.png")


# ── Plot 2: Amount distribution ────────────────────────────────────────────
def plot_amount_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Transaction Amount by Class", fontsize=13, fontweight="bold")

    for ax, log_scale in zip(axes, [False, True]):
        for cls, label, color in [(0, "Legitimate", PALETTE["legit"]), (1, "Fraud", PALETTE["fraud"])]:
            data = df[df["Class"] == cls]["Amount"]
            ax.hist(data, bins=60, alpha=0.65, label=label, color=color,
                    log=log_scale, density=True, edgecolor="none")
        ax.set_xlabel("Amount (USD)")
        ax.set_ylabel("Density" + (" (log)" if log_scale else ""))
        ax.set_title(f"{'Log-scale' if log_scale else 'Linear-scale'} distribution")
        ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_amount_distribution.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 02_amount_distribution.png")


# ── Plot 3: Time of day pattern ────────────────────────────────────────────
def plot_time_pattern(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    for cls, label, color in [(0, "Legitimate", PALETTE["legit"]), (1, "Fraud", PALETTE["fraud"])]:
        data = df[df["Class"] == cls]["Time"]
        ax.hist(data / 3600, bins=48, alpha=0.6, label=label, color=color,
                density=True, edgecolor="none")
    ax.set_xlabel("Time (hours from start)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction time distribution by class", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_time_pattern.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 03_time_pattern.png")


# ── Plot 4: PCA feature distributions (top 6 most discriminative) ─────────
def plot_pca_features(df):
    # Find most discriminative features by mean difference
    v_cols = [f"V{i}" for i in range(1, 29)]
    fraud  = df[df["Class"] == 1][v_cols]
    legit  = df[df["Class"] == 0][v_cols]
    delta  = (fraud.mean() - legit.mean()).abs().sort_values(ascending=False)
    top6   = delta.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Top 6 Most Discriminative PCA Features", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, col in zip(axes, top6):
        for cls, label, color in [(0, "Legitimate", PALETTE["legit"]), (1, "Fraud", PALETTE["fraud"])]:
            data = df[df["Class"] == cls][col]
            ax.hist(data, bins=50, alpha=0.6, label=label, color=color,
                    density=True, edgecolor="none")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_pca_features.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 04_pca_features.png")


# ── Plot 5: Correlation heatmap ────────────────────────────────────────────
def plot_correlation(df):
    v_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    corr   = df[v_cols].corr()["Class"].drop("Class").sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE["fraud"] if v > 0 else PALETTE["legit"] for v in corr.values]
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor="none")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson correlation with Class label")
    ax.set_title("Feature Correlation with Fraud Label", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_correlation.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 05_feature_correlation.png")


# ── Plot 6: Box plots for top features ────────────────────────────────────
def plot_boxplots(df):
    v_cols = [f"V{i}" for i in range(1, 29)]
    delta  = (df[df["Class"]==1][v_cols].mean() - df[df["Class"]==0][v_cols].mean()).abs()
    top8   = delta.sort_values(ascending=False).head(8).index.tolist()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    fig.suptitle("Box Plots — Top 8 Discriminative Features", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, col in zip(axes, top8):
        data_legit = df[df["Class"]==0][col].sample(5000, random_state=42)
        data_fraud = df[df["Class"]==1][col]
        ax.boxplot([data_legit, data_fraud],
                   labels=["Legit", "Fraud"],
                   patch_artist=True,
                   boxprops=dict(facecolor="none"),
                   medianprops=dict(color="black", linewidth=2),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3))
        parts = ax.patches
        if len(parts) >= 2:
            parts[0].set_facecolor(PALETTE["legit"] + "55")
            parts[1].set_facecolor(PALETTE["fraud"] + "55")
        ax.set_title(col, fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_boxplots.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 06_boxplots.png")


# ── Summary stats ──────────────────────────────────────────────────────────
def print_summary(df):
    print("\n" + "="*60)
    print("  DATASET SUMMARY")
    print("="*60)
    print(f"  Total transactions : {len(df):>10,}")
    fraud = df["Class"].sum()
    print(f"  Legitimate         : {len(df)-fraud:>10,}  ({(1-df['Class'].mean()):.4%})")
    print(f"  Fraudulent         : {fraud:>10,}  ({df['Class'].mean():.4%})")
    print(f"  Features           : {df.shape[1]-1:>10}")
    print(f"  Missing values     : {df.isnull().sum().sum():>10}")
    print(f"  Avg legit amount   : ${df[df['Class']==0]['Amount'].mean():>9.2f}")
    print(f"  Avg fraud amount   : ${df[df['Class']==1]['Amount'].mean():>9.2f}")
    print(f"  Time span (hours)  : {df['Time'].max()/3600:>10.1f}")
    print("="*60 + "\n")


def main():
    df = load_data()
    print_summary(df)
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_time_pattern(df)
    plot_pca_features(df)
    plot_correlation(df)
    plot_boxplots(df)
    print(f"\n[DONE] All EDA plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
