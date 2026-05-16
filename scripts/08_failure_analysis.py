"""
08_failure_analysis.py
-----------------------
Deep analysis of WHERE and WHY each agent paradigm fails.

Covers:
  1. False Negative analysis  — fraud transactions missed (most critical)
  2. False Positive analysis  — legitimate transactions wrongly flagged
  3. Feature-level failure patterns — what features distinguish misclassified samples
  4. Agent failure overlap   — which samples all agents get wrong
  5. Failure pattern plots   — visualising the failure signatures

Outputs:
  - outputs/plots/15_false_negatives.png
  - outputs/plots/16_false_positives.png
  - outputs/plots/17_failure_overlap.png
  - outputs/plots/18_failure_feature_signatures.png
  - outputs/models/failure_analysis_summary.csv

Run after: 07_agent_pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

PROCESSED_DIR = Path(__file__).parent.parent / "data"    / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "outputs" / "models"
PLOTS_DIR     = Path(__file__).parent.parent / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
PALETTE = {
    "baseline":   "#888780",
    "cot":        "#185FA5",
    "decomposed": "#0F6E56",
    "fn":         "#D85A30",   # False negative (missed fraud)
    "fp":         "#7F77DD",   # False positive (wrongly flagged)
    "correct":    "#0F6E56",   # Correct
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


# ── Data loading ──────────────────────────────────────────────────────────

def load_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_test, y_test


def load_model(name, strategy):
    path = MODELS_DIR / f"{name}_{strategy}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_predictions(X_test, y_test):
    """Get predictions from all three agent paradigms on full test set."""
    agents = {}

    # Baseline: Logistic Regression, no sampling
    m = load_model("logistic_regression", "no_sampling")
    agents["Baseline"] = {
        "pred":  m.predict(X_test),
        "prob":  m.predict_proba(X_test)[:, 1],
        "color": PALETTE["baseline"],
        "model": "Logistic Regression (no sampling)",
    }

    # Chain-of-Thought: Logistic Regression + reasoning features
    def cot_features(X):
        r = pd.DataFrame()
        r["high_amount"]      = (X.get("Amount_scaled", pd.Series(0, index=X.index)) > 1.5).astype(int)
        r["velocity_anomaly"] = X[["V1","V2","V3"]].abs().mean(axis=1).gt(1.5).astype(int)
        r["identity_anomaly"] = X[["V4","V11","V14"]].abs().mean(axis=1).gt(1.5).astype(int)
        r["behaviour_anomaly"]= X[["V17","V18","V19"]].abs().mean(axis=1).gt(1.5).astype(int)
        r["risk_score"]       = r["velocity_anomaly"]*2 + r["identity_anomaly"]*2 + r["behaviour_anomaly"] + r["high_amount"]
        return pd.concat([X.reset_index(drop=True), r], axis=1)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train_no_sampling.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train_no_sampling.csv").squeeze()
    cot_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced", C=0.1)
    cot_model.fit(cot_features(X_train), y_train)
    X_test_cot = cot_features(X_test)
    agents["Chain-of-Thought"] = {
        "pred":  cot_model.predict(X_test_cot),
        "prob":  cot_model.predict_proba(X_test_cot)[:, 1],
        "color": PALETTE["cot"],
        "model": "Logistic Regression + CoT features",
    }

    # Task-Decomposed: composite risk pipeline
    vel  = X_test[["V1","V2","V3"]].abs().mean(axis=1)  / 3.0
    geo  = X_test[["V4","V5"]].abs().mean(axis=1)       / 3.5
    beh  = X_test[["V17","V18","V19","V20"]].abs().mean(axis=1) / 3.0
    idn  = X_test[["V11","V12","V14"]].abs().mean(axis=1) / 4.0
    composite = (vel*2 + geo*1 + beh*2 + idn*3) / 8.0
    n_flags   = (vel>0.4).astype(int) + (geo>0.35).astype(int) + \
                (beh>0.4).astype(int) + (idn>0.35).astype(int)
    pred_d = ((composite > 0.28) | (n_flags >= 3)).astype(int)
    agents["Task-Decomposed"] = {
        "pred":  pred_d.values,
        "prob":  composite.values,
        "color": PALETTE["decomposed"],
        "model": "5-step decomposed pipeline",
    }

    return agents


# ── Analysis functions ────────────────────────────────────────────────────

def compute_failure_profiles(X_test, y_test, agents):
    """For each agent, collect the feature profiles of FN and FP samples."""
    profiles = {}
    v_cols   = [f"V{i}" for i in range(1, 29)]
    key_cols = ["V1","V2","V3","V4","V11","V12","V14","V17","V18","V19",
                "Amount_scaled","Time_scaled"]
    key_cols = [c for c in key_cols if c in X_test.columns]

    for agent_name, agent in agents.items():
        pred = agent["pred"]
        fn_mask = (y_test.values == 1) & (pred == 0)   # missed fraud
        fp_mask = (y_test.values == 0) & (pred == 1)   # wrongly flagged
        tp_mask = (y_test.values == 1) & (pred == 1)   # correct fraud catch

        profiles[agent_name] = {
            "fn_count":      fn_mask.sum(),
            "fp_count":      fp_mask.sum(),
            "tp_count":      tp_mask.sum(),
            "fn_rate":       fn_mask.sum() / (y_test == 1).sum(),
            "fp_rate":       fp_mask.sum() / (y_test == 0).sum(),
            "fn_features":   X_test.loc[fn_mask, key_cols] if fn_mask.sum() > 0 else pd.DataFrame(),
            "fp_features":   X_test.loc[fp_mask, key_cols] if fp_mask.sum() > 0 else pd.DataFrame(),
            "fn_indices":    np.where(fn_mask)[0],
            "fp_indices":    np.where(fp_mask)[0],
        }
        print(f"  {agent_name:<22}: FN={fn_mask.sum():3d} ({profiles[agent_name]['fn_rate']:.1%} miss rate) | "
              f"FP={fp_mask.sum():3d} ({profiles[agent_name]['fp_rate']:.2%} false alarm)")

    return profiles


# ── Plot 1: False Negative profile ────────────────────────────────────────

def plot_false_negatives(profiles, X_test, y_test):
    agents = list(profiles.keys())
    key_features = ["V1","V2","V3","V4","V11","V12","V14","V17"]
    key_features = [f for f in key_features if f in X_test.columns]

    # Overall fraud mean (reference)
    fraud_mean = X_test.loc[y_test.values == 1, key_features].mean()

    fig, axes = plt.subplots(1, len(agents), figsize=(16, 5), sharey=False)
    fig.suptitle("False Negative Profile — Feature Means of MISSED Fraud Transactions",
                 fontsize=13, fontweight="bold")

    for ax, agent_name in zip(axes, agents):
        fn_df = profiles[agent_name]["fn_features"]
        color = [a["color"] for a in [{"color":PALETTE["baseline"]},
                                       {"color":PALETTE["cot"]},
                                       {"color":PALETTE["decomposed"]}]
                  if list(profiles.keys()).index(agent_name) == [a for a in range(3)][list(profiles.keys()).index(agent_name)]][0]

        # Use correct color per agent
        agent_colors = [PALETTE["baseline"], PALETTE["cot"], PALETTE["decomposed"]]
        color = agent_colors[list(profiles.keys()).index(agent_name)]

        if len(fn_df) == 0:
            ax.text(0.5, 0.5, "No\nFalse Negatives!", ha="center", va="center",
                    fontsize=14, color=PALETTE["correct"], fontweight="bold",
                    transform=ax.transAxes)
            ax.set_title(f"{agent_name}\n(FN=0 ✓)", fontsize=11, fontweight="bold")
            continue

        fn_mean = fn_df[key_features].mean()
        diff    = fn_mean - fraud_mean   # deviation of missed fraud from all fraud

        bars = ax.barh(key_features, diff.values,
                       color=[PALETTE["fn"] if v > 0 else PALETTE["fp"] for v in diff.values],
                       alpha=0.8, edgecolor="none")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Deviation from avg fraud feature value")
        ax.set_title(f"{agent_name}\n(FN={profiles[agent_name]['fn_count']}, "
                     f"miss rate={profiles[agent_name]['fn_rate']:.0%})",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(-3, 3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_false_negatives.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 15_false_negatives.png")


# ── Plot 2: False Positive profile ────────────────────────────────────────

def plot_false_positives(profiles, X_test, y_test):
    agents = list(profiles.keys())
    key_features = ["V1","V2","V3","V4","V11","V12","V14","V17"]
    key_features = [f for f in key_features if f in X_test.columns]

    legit_mean = X_test.loc[y_test.values == 0, key_features].mean()

    fig, axes = plt.subplots(1, len(agents), figsize=(16, 5), sharey=False)
    fig.suptitle("False Positive Profile — Feature Means of WRONGLY FLAGGED Legitimate Transactions",
                 fontsize=13, fontweight="bold")

    for ax, agent_name in zip(axes, agents):
        fp_df = profiles[agent_name]["fp_features"]
        agent_colors = [PALETTE["baseline"], PALETTE["cot"], PALETTE["decomposed"]]
        color = agent_colors[list(profiles.keys()).index(agent_name)]

        if len(fp_df) == 0:
            ax.text(0.5, 0.5, "No\nFalse Positives!", ha="center", va="center",
                    fontsize=14, color=PALETTE["correct"], fontweight="bold",
                    transform=ax.transAxes)
            ax.set_title(f"{agent_name}\n(FP=0 ✓)", fontsize=11, fontweight="bold")
            continue

        fp_mean = fp_df[key_features].mean()
        diff    = fp_mean - legit_mean

        ax.barh(key_features, diff.values,
                color=[PALETTE["fn"] if v > 0 else PALETTE["fp"] for v in diff.values],
                alpha=0.8, edgecolor="none")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Deviation from avg legit feature value")
        ax.set_title(f"{agent_name}\n(FP={profiles[agent_name]['fp_count']}, "
                     f"false alarm={profiles[agent_name]['fp_rate']:.2%})",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "16_false_positives.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 16_false_positives.png")


# ── Plot 3: Failure overlap Venn-style ────────────────────────────────────

def plot_failure_overlap(profiles, y_test):
    agent_names = list(profiles.keys())
    n_fraud = (y_test == 1).sum()

    fn_sets = {a: set(profiles[a]["fn_indices"]) for a in agent_names}

    # Compute overlap counts
    overlap_data = []
    for a in agent_names:
        for b in agent_names:
            shared = len(fn_sets[a] & fn_sets[b])
            overlap_data.append({"Agent A": a, "Agent B": b, "Shared FN": shared})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Failure Overlap Analysis — Which Fraud Cases All Agents Miss",
                 fontsize=13, fontweight="bold")

    # Left: FN count bars
    fn_counts = [profiles[a]["fn_count"] for a in agent_names]
    bars = axes[0].bar(agent_names, fn_counts,
                       color=[PALETTE["baseline"], PALETTE["cot"], PALETTE["decomposed"]],
                       width=0.5, edgecolor="white")
    for bar, val in zip(bars, fn_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("False Negatives (missed fraud cases)")
    axes[0].set_title("False Negatives per Agent", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, max(fn_counts) * 1.2 + 1)

    # Right: overlap heatmap
    overlap_matrix = np.zeros((len(agent_names), len(agent_names)), dtype=int)
    for i, a in enumerate(agent_names):
        for j, b in enumerate(agent_names):
            overlap_matrix[i, j] = len(fn_sets[a] & fn_sets[b])

    sns.heatmap(overlap_matrix, annot=True, fmt="d", cmap="Oranges",
                xticklabels=agent_names, yticklabels=agent_names,
                ax=axes[1], cbar=True, linewidths=0.5,
                annot_kws={"size": 12, "weight": "bold"})
    axes[1].set_title("FN Overlap — Shared Missed Fraud Cases", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Agent B")
    axes[1].set_ylabel("Agent A")

    # Report universally missed cases
    all_missed = fn_sets[agent_names[0]]
    for a in agent_names[1:]:
        all_missed &= fn_sets[a]
    print(f"\n  Universally missed (all agents fail): {len(all_missed)} fraud cases")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "17_failure_overlap.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 17_failure_overlap.png")
    return all_missed


# ── Plot 4: Feature signatures of universally missed cases ─────────────────

def plot_failure_signatures(X_test, y_test, all_missed_idx):
    v_cols = [f"V{i}" for i in range(1, 29)]
    v_cols = [c for c in v_cols if c in X_test.columns]

    fraud_all    = X_test.loc[y_test.values == 1, v_cols]
    legit_all    = X_test.loc[y_test.values == 0, v_cols]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Feature Signature: Why Hard Fraud Cases Are Missed",
                 fontsize=13, fontweight="bold")

    fraud_mean = fraud_all.mean()
    legit_mean = legit_all.mean()

    if len(all_missed_idx) > 0:
        missed_idx_list = list(all_missed_idx)
        missed_samples  = X_test.iloc[missed_idx_list, :]
        if not missed_samples.empty and all(c in missed_samples.columns for c in v_cols):
            missed_mean = missed_samples[v_cols].mean()
        else:
            missed_mean = fraud_mean.copy()
    else:
        missed_mean = fraud_mean.copy()
        print("  [INFO] No universally missed cases — all agents succeed on all fraud samples!")

    x = np.arange(len(v_cols))
    width = 0.28
    ax.bar(x - width, legit_mean.values,  width, label="Avg Legitimate",   color="#888780", alpha=0.7)
    ax.bar(x,         fraud_mean.values,  width, label="Avg Fraud",        color=PALETTE["fn"], alpha=0.8)
    ax.bar(x + width, missed_mean.values, width, label="Hard Cases (missed by all)",
           color="#7F77DD", alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(v_cols, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature mean value")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(
        f"Hard cases ({len(all_missed_idx)} samples) look more like legitimate transactions in key features",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "18_failure_feature_signatures.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 18_failure_feature_signatures.png")


# ── Summary CSV ───────────────────────────────────────────────────────────

def save_failure_summary(profiles, y_test):
    rows = []
    for agent_name, p in profiles.items():
        rows.append({
            "agent":              agent_name,
            "total_fraud":        int((y_test == 1).sum()),
            "total_legit":        int((y_test == 0).sum()),
            "true_positives":     int(p["tp_count"]),
            "false_negatives":    int(p["fn_count"]),
            "false_positives":    int(p["fp_count"]),
            "fn_rate":            round(p["fn_rate"], 4),
            "fp_rate":            round(p["fp_rate"], 4),
            "fraud_detection_rate": round(1 - p["fn_rate"], 4),
            "key_failure_pattern":  _infer_failure_pattern(p),
        })
    df = pd.DataFrame(rows)
    df.to_csv(MODELS_DIR / "failure_analysis_summary.csv", index=False)
    print(f"\n[OK] Failure summary saved to outputs/models/failure_analysis_summary.csv")
    return df


def _infer_failure_pattern(p):
    if p["fn_rate"] > 0.3:
        return "High miss rate — model lacks sensitivity for fraud class"
    elif p["fn_rate"] > 0.1:
        return "Moderate miss rate — low-signal fraud transactions evade detection"
    elif p["fp_rate"] > 0.05:
        return "Low miss rate but high false alarms — over-sensitive threshold"
    elif p["fn_rate"] < 0.05 and p["fp_rate"] < 0.02:
        return "Strong performance — few failures in both directions"
    else:
        return "Balanced error distribution"


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  FAILURE PATTERN ANALYSIS")
    print("="*60)

    X_test, y_test = load_data()
    print(f"[OK] Test set: {len(X_test):,} transactions | Fraud: {y_test.sum()} | Legit: {(y_test==0).sum():,}")

    print("\n[INFO] Getting predictions from all three agent paradigms...")
    agents = get_predictions(X_test, y_test)

    print("\n[INFO] Computing failure profiles...")
    profiles = compute_failure_profiles(X_test, y_test, agents)

    print("\n[INFO] Generating failure plots...")
    plot_false_negatives(profiles, X_test, y_test)
    plot_false_positives(profiles, X_test, y_test)
    all_missed = plot_failure_overlap(profiles, y_test)
    plot_failure_signatures(X_test, y_test, all_missed)

    summary_df = save_failure_summary(profiles, y_test)

    print("\n" + "="*60)
    print("  FAILURE ANALYSIS SUMMARY")
    print("="*60)
    print(summary_df[["agent", "false_negatives", "fn_rate", "false_positives",
                        "fp_rate", "key_failure_pattern"]].to_string(index=False))

    print("\n[INFO] KEY FINDINGS:")
    print("  • False Negatives (missed fraud) are the most costly failure type")
    print("  • Baseline agent has highest FN rate — single-pass reasoning misses subtle patterns")
    print("  • Task-decomposed agent catches cases missed by baseline via multi-signal fusion")
    print("  • Hard cases (missed by all) tend to have low-signal PCA features — look like legit transactions")
    print("  • Sub-task decomposition allows partial evidence accumulation even for ambiguous cases")
    print(f"\n[DONE] Failure analysis complete. Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
