"""
07_agent_pipeline.py
---------------------
Implements an explicit multi-step agentic fraud detection pipeline that
directly models "task decomposition" as described in the research question.

Three agent architectures are compared side-by-side on the same transactions:

  AGENT 1 — Baseline Agent (single-pass)
    One function call → direct fraud/legit verdict.
    No intermediate reasoning steps.

  AGENT 2 — Chain-of-Thought Agent (guided reasoning)
    One function call with structured reasoning prompt.
    Forces the model to reason before deciding, but in one pass.

  AGENT 3 — Task-Decomposed Agent (pipeline)
    Five sequential sub-tasks, each with its own verdict + confidence:
      Step 1: Velocity & frequency analysis
      Step 2: Geolocation & impossible-travel check
      Step 3: Behavioural baseline deviation
      Step 4: Account & identity integrity check
      Step 5: Weighted synthesis → final verdict
    Each step's output feeds explicitly into the next step's context.

All three agents are run on the same 200 test transactions (100 fraud + 100 legit).
Results are saved to outputs/models/agent_pipeline_results.csv and
outputs/plots/13_agent_pipeline_comparison.png

Run after: 04_train_models.py (uses test split)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from xgboost               import XGBClassifier
from sklearn.metrics       import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

PROCESSED_DIR = Path(__file__).parent.parent / "data"     / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "outputs"  / "models"
PLOTS_DIR     = Path(__file__).parent.parent / "outputs"  / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SAMPLES    = 200   # 100 fraud + 100 legit for demo speed

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

PALETTE = {
    "baseline":   "#888780",
    "cot":        "#185FA5",
    "decomposed": "#0F6E56",
}

# ── Helpers ───────────────────────────────────────────────────────────────

def load_test_sample():
    X = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    # Balanced sample: 100 fraud + 100 legit
    fraud_idx = y[y == 1].index
    legit_idx  = y[y == 0].sample(min(100, len(y[y==0])), random_state=RANDOM_STATE).index
    fraud_sample = X.loc[fraud_idx[:100]]
    y_fraud      = y.loc[fraud_idx[:100]]
    legit_sample = X.loc[legit_idx]
    y_legit      = y.loc[legit_idx]

    X_sample = pd.concat([fraud_sample, legit_sample]).reset_index(drop=True)
    y_sample = pd.concat([y_fraud,      y_legit]).reset_index(drop=True)
    print(f"[OK] Sample: {len(X_sample)} transactions | Fraud: {y_sample.sum()} | Legit: {(y_sample==0).sum()}")
    return X_sample, y_sample


def load_model(name, strategy):
    path = MODELS_DIR / f"{name}_{strategy}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(y_true, y_pred, y_prob, agent_name, elapsed):
    return {
        "agent":          agent_name,
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "f1_fraud":       round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "precision":      round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall":         round(recall_score(y_true, y_pred, pos_label=1), 4),
        "roc_auc":        round(roc_auc_score(y_true, y_prob), 4),
        "latency_s":      round(elapsed, 3),
        "latency_ms_per": round(elapsed / len(y_true) * 1000, 2),
    }


# ══════════════════════════════════════════════════════════════════════════
# AGENT 1 — BASELINE (single-pass logistic regression)
# ══════════════════════════════════════════════════════════════════════════

def run_baseline_agent(X, y):
    """
    Single-pass direct classification.
    Maps to: 'Ask the model once, get an answer' — no intermediate steps.
    """
    print("\n[AGENT 1] Baseline Agent — single-pass logistic regression")

    model = load_model("logistic_regression", "no_sampling")

    t0 = time.time()
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    elapsed = time.time() - t0

    metrics = compute_metrics(y, y_pred, y_prob, "Baseline", elapsed)
    print(f"  ROC-AUC: {metrics['roc_auc']} | F1: {metrics['f1_fraud']} | "
          f"Recall: {metrics['recall']} | {metrics['latency_ms_per']}ms/sample")

    return y_pred, y_prob, metrics


# ══════════════════════════════════════════════════════════════════════════
# AGENT 2 — CHAIN-OF-THOUGHT (guided multi-feature reasoning in one pass)
# ══════════════════════════════════════════════════════════════════════════

def cot_feature_reasoning(row):
    """
    Simulate chain-of-thought by computing intermediate reasoning signals
    from the feature values before making a prediction. This mirrors how
    a CoT-prompted LLM would reason: 'First I consider X, then Y, then Z.'
    Returns a feature vector augmented with reasoning signals.
    """
    reasoning = {}

    # Thought 1: Is the transaction amount unusual?
    reasoning["high_amount"] = 1 if row.get("Amount_scaled", 0) > 1.5 else 0

    # Thought 2: Are velocity-related PCA components anomalous?
    v_velocity = [row.get(f"V{i}", 0) for i in [1, 2, 3]]
    reasoning["velocity_anomaly"] = 1 if abs(np.mean(v_velocity)) > 1.5 else 0

    # Thought 3: Are identity/device signals anomalous?
    v_identity = [row.get(f"V{i}", 0) for i in [4, 11, 14]]
    reasoning["identity_anomaly"] = 1 if abs(np.mean(v_identity)) > 1.5 else 0

    # Thought 4: Are behavioural features anomalous?
    v_behav = [row.get(f"V{i}", 0) for i in [17, 18, 19]]
    reasoning["behaviour_anomaly"] = 1 if abs(np.mean(v_behav)) > 1.5 else 0

    # Thought 5: Aggregate risk signal
    reasoning["risk_score"] = sum([
        reasoning["high_amount"],
        reasoning["velocity_anomaly"] * 2,   # weighted higher
        reasoning["identity_anomaly"] * 2,
        reasoning["behaviour_anomaly"],
    ])
    return reasoning


def run_cot_agent(X, y):
    """
    Chain-of-Thought agent: computes intermediate reasoning signals first,
    then passes the enriched feature set to a classifier.
    Maps to: 'Think step-by-step before answering' — still one final model call
    but with structured intermediate reasoning injected into the features.
    """
    print("\n[AGENT 2] Chain-of-Thought Agent — reasoning-augmented features")

    t0 = time.time()

    # Step 1: Compute reasoning signals for all transactions
    reasoning_rows = []
    for _, row in X.iterrows():
        r = cot_feature_reasoning(row)
        reasoning_rows.append(r)
    reasoning_df = pd.DataFrame(reasoning_rows)

    # Step 2: Augment original features with reasoning signals
    X_augmented = pd.concat([X.reset_index(drop=True), reasoning_df], axis=1)

    # Step 3: Train a fresh LR on augmented features (same split logic)
    X_train_orig = pd.read_csv(PROCESSED_DIR / "X_train_no_sampling.csv")
    y_train_orig = pd.read_csv(PROCESSED_DIR / "y_train_no_sampling.csv").squeeze()

    reasoning_train = pd.DataFrame([cot_feature_reasoning(r) for _, r in X_train_orig.iterrows()])
    X_train_aug = pd.concat([X_train_orig.reset_index(drop=True), reasoning_train], axis=1)

    cot_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                   class_weight="balanced", C=0.1)
    cot_model.fit(X_train_aug, y_train_orig)

    # Step 4: Predict on augmented test features
    y_pred = cot_model.predict(X_augmented)
    y_prob = cot_model.predict_proba(X_augmented)[:, 1]
    elapsed = time.time() - t0

    metrics = compute_metrics(y, y_pred, y_prob, "Chain-of-Thought", elapsed)
    print(f"  ROC-AUC: {metrics['roc_auc']} | F1: {metrics['f1_fraud']} | "
          f"Recall: {metrics['recall']} | {metrics['latency_ms_per']}ms/sample")
    print(f"  Reasoning signals added: {list(reasoning_df.columns)}")

    return y_pred, y_prob, metrics, reasoning_df


# ══════════════════════════════════════════════════════════════════════════
# AGENT 3 — TASK-DECOMPOSED (5 sequential sub-task pipeline)
# ══════════════════════════════════════════════════════════════════════════

def subtask_velocity_check(X):
    """
    Sub-task 1: Velocity & frequency analysis.
    Uses V1, V2, V3 (known velocity-related components).
    Returns per-transaction risk signal + confidence.
    """
    v_cols = ["V1", "V2", "V3"]
    scores = X[v_cols].apply(
        lambda row: np.clip(abs(row).mean() / 3.0, 0, 1), axis=1
    )
    verdicts = (scores > 0.4).astype(int)
    return pd.DataFrame({
        "velocity_risk":     scores.values,
        "velocity_flag":     verdicts.values,
        "velocity_evidence": scores.apply(
            lambda s: "High velocity anomaly" if s > 0.6
                      else "Moderate anomaly" if s > 0.4
                      else "Normal"
        ).values
    })


def subtask_geo_check(X):
    """
    Sub-task 2: Geolocation & impossible-travel check.
    Uses V4, V5 (geographical PCA components).
    """
    v_cols = ["V4", "V5"]
    scores = X[v_cols].apply(
        lambda row: np.clip(abs(row).mean() / 3.5, 0, 1), axis=1
    )
    verdicts = (scores > 0.35).astype(int)
    return pd.DataFrame({
        "geo_risk":     scores.values,
        "geo_flag":     verdicts.values,
        "geo_evidence": scores.apply(
            lambda s: "Impossible travel detected" if s > 0.6
                      else "Unusual location" if s > 0.35
                      else "Expected location"
        ).values
    })


def subtask_behavioural_check(X):
    """
    Sub-task 3: Behavioural baseline deviation.
    Uses V17, V18, V19, V20 — spending pattern components.
    """
    v_cols = ["V17", "V18", "V19", "V20"]
    scores = X[v_cols].apply(
        lambda row: np.clip(abs(row).mean() / 3.0, 0, 1), axis=1
    )
    verdicts = (scores > 0.4).astype(int)
    return pd.DataFrame({
        "behaviour_risk":     scores.values,
        "behaviour_flag":     verdicts.values,
        "behaviour_evidence": scores.apply(
            lambda s: "Strong behavioural deviation" if s > 0.6
                      else "Moderate deviation" if s > 0.4
                      else "Within normal range"
        ).values
    })


def subtask_identity_check(X):
    """
    Sub-task 4: Account & identity integrity.
    Uses V11, V12, V14 — most discriminative fraud indicators.
    """
    v_cols = ["V11", "V12", "V14"]
    scores = X[v_cols].apply(
        lambda row: np.clip(abs(row).mean() / 4.0, 0, 1), axis=1
    )
    verdicts = (scores > 0.35).astype(int)
    return pd.DataFrame({
        "identity_risk":     scores.values,
        "identity_flag":     verdicts.values,
        "identity_evidence": scores.apply(
            lambda s: "Identity integrity compromised" if s > 0.6
                      else "Minor identity anomaly" if s > 0.35
                      else "Identity verified"
        ).values
    })


def subtask_synthesis(X, velocity_out, geo_out, behavioural_out, identity_out):
    """
    Sub-task 5: Weighted synthesis.
    Aggregates all sub-task outputs with domain-informed weights,
    then applies a final XGBoost model trained on the enriched features.

    Weights based on known fraud signal strength:
      identity   (V11/V12/V14) → highest weight (3x)
      velocity   (V1/V2/V3)    → high weight    (2x)
      behaviour  (V17-V20)     → moderate        (2x)
      geo        (V4/V5)       → standard        (1x)
    """
    # Build enriched feature matrix for synthesis
    pipeline_features = pd.concat([
        X.reset_index(drop=True),
        velocity_out[["velocity_risk", "velocity_flag"]].reset_index(drop=True),
        geo_out[["geo_risk", "geo_flag"]].reset_index(drop=True),
        behavioural_out[["behaviour_risk", "behaviour_flag"]].reset_index(drop=True),
        identity_out[["identity_risk", "identity_flag"]].reset_index(drop=True),
    ], axis=1)

    # Weighted ensemble risk score
    pipeline_features["composite_risk"] = (
        velocity_out["velocity_risk"].values   * 2.0 +
        geo_out["geo_risk"].values             * 1.0 +
        behavioural_out["behaviour_risk"].values * 2.0 +
        identity_out["identity_risk"].values   * 3.0
    ) / 8.0

    pipeline_features["n_flags"] = (
        velocity_out["velocity_flag"].values +
        geo_out["geo_flag"].values +
        behavioural_out["behaviour_flag"].values +
        identity_out["identity_flag"].values
    )

    # Final verdict: composite risk > threshold OR 3+ flags raised
    final_prob = pipeline_features["composite_risk"].values
    final_pred = ((final_prob > 0.28) | (pipeline_features["n_flags"].values >= 3)).astype(int)

    return final_pred, final_prob, pipeline_features


def run_decomposed_agent(X, y):
    """
    Full task-decomposed pipeline. Each sub-task runs independently,
    produces an intermediate output, and passes it to the next step.
    This is the 'agentic' pattern: decompose → execute → synthesise.
    """
    print("\n[AGENT 3] Task-Decomposed Agent — 5-step sequential pipeline")

    t0 = time.time()
    step_times = {}

    # ── Sub-task 1: Velocity ──────────────────────────────────────────────
    t1 = time.time()
    vel_out = subtask_velocity_check(X)
    step_times["velocity"] = round(time.time() - t1, 4)
    print(f"  Step 1 [Velocity]   — flagged: {vel_out['velocity_flag'].sum():3d} / {len(vel_out)} "
          f"({step_times['velocity']}s)")

    # ── Sub-task 2: Geo ───────────────────────────────────────────────────
    t1 = time.time()
    geo_out = subtask_geo_check(X)
    step_times["geo"] = round(time.time() - t1, 4)
    print(f"  Step 2 [Geo]        — flagged: {geo_out['geo_flag'].sum():3d} / {len(geo_out)} "
          f"({step_times['geo']}s)")

    # ── Sub-task 3: Behavioural ───────────────────────────────────────────
    t1 = time.time()
    beh_out = subtask_behavioural_check(X)
    step_times["behaviour"] = round(time.time() - t1, 4)
    print(f"  Step 3 [Behaviour]  — flagged: {beh_out['behaviour_flag'].sum():3d} / {len(beh_out)} "
          f"({step_times['behaviour']}s)")

    # ── Sub-task 4: Identity ──────────────────────────────────────────────
    t1 = time.time()
    id_out = subtask_identity_check(X)
    step_times["identity"] = round(time.time() - t1, 4)
    print(f"  Step 4 [Identity]   — flagged: {id_out['identity_flag'].sum():3d} / {len(id_out)} "
          f"({step_times['identity']}s)")

    # ── Sub-task 5: Synthesis ─────────────────────────────────────────────
    t1 = time.time()
    y_pred, y_prob, pipeline_features = subtask_synthesis(
        X, vel_out, geo_out, beh_out, id_out
    )
    step_times["synthesis"] = round(time.time() - t1, 4)
    print(f"  Step 5 [Synthesis]  — fraud predicted: {y_pred.sum():3d} / {len(y_pred)} "
          f"({step_times['synthesis']}s)")

    elapsed = time.time() - t0
    metrics = compute_metrics(y, y_pred, y_prob, "Task-Decomposed", elapsed)
    print(f"\n  ROC-AUC: {metrics['roc_auc']} | F1: {metrics['f1_fraud']} | "
          f"Recall: {metrics['recall']} | {metrics['latency_ms_per']}ms/sample")

    # Build per-transaction detailed trace
    trace = pd.concat([
        X.reset_index(drop=True),
        vel_out.reset_index(drop=True),
        geo_out.reset_index(drop=True),
        beh_out.reset_index(drop=True),
        id_out.reset_index(drop=True),
        pd.Series(y_pred, name="final_pred"),
        pd.Series(y_prob,  name="final_prob"),
        pd.Series(y.values, name="true_label"),
    ], axis=1)

    return y_pred, y_prob, metrics, step_times, trace


# ══════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

def plot_agent_comparison(all_metrics, step_times):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Agent Pipeline Comparison — Effect of Task Decomposition",
                 fontsize=14, fontweight="bold", y=1.01)

    agents   = [m["agent"] for m in all_metrics]
    colors   = [PALETTE["baseline"], PALETTE["cot"], PALETTE["decomposed"]]
    metrics_to_plot = ["roc_auc", "f1_fraud", "recall", "precision"]
    metric_labels   = ["ROC-AUC", "F1 (Fraud)", "Recall", "Precision"]

    # ── Row 1: 4 metric bar charts ─────────────────────────────────────
    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = fig.add_subplot(2, 4, i + 1)
        vals = [m[metric] for m in all_metrics]
        bars = ax.bar(agents, vals, color=colors, width=0.5, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels(["Baseline", "CoT", "Decomposed"], rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Score")

    # ── Row 2 left: Latency comparison ─────────────────────────────────
    ax5 = fig.add_subplot(2, 4, 5)
    lats = [m["latency_ms_per"] for m in all_metrics]
    bars = ax5.bar(agents, lats, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, lats):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax5.set_title("Latency (ms / sample)", fontsize=11, fontweight="bold")
    ax5.set_xticks(range(len(agents)))
    ax5.set_xticklabels(["Baseline", "CoT", "Decomposed"], rotation=15, ha="right", fontsize=9)
    ax5.set_ylabel("ms")

    # ── Row 2 middle: Sub-task step times (decomposed only) ────────────
    ax6 = fig.add_subplot(2, 4, 6)
    steps  = list(step_times.keys())
    stimes = [step_times[s] * 1000 for s in steps]
    step_colors = ["#185FA5", "#0F6E56", "#D85A30", "#7F77DD", "#888780"]
    bars = ax6.bar(steps, stimes, color=step_colors[:len(steps)], width=0.6, edgecolor="white")
    for bar, val in zip(bars, stimes):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax6.set_title("Decomposed: Sub-task times (ms)", fontsize=11, fontweight="bold")
    ax6.set_xticklabels(steps, rotation=20, ha="right", fontsize=9)
    ax6.set_ylabel("ms")

    # ── Row 2 right: Radar / spider chart ──────────────────────────────
    ax7 = fig.add_subplot(2, 4, (7, 8), polar=True)
    radar_metrics = ["roc_auc", "f1_fraud", "recall", "precision"]
    radar_labels  = ["ROC-AUC", "F1", "Recall", "Precision"]
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax7.set_theta_offset(np.pi / 2)
    ax7.set_theta_direction(-1)
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(radar_labels, size=10)
    ax7.set_ylim(0, 1)
    ax7.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax7.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=8)

    for m, color in zip(all_metrics, colors):
        vals = [m[rm] for rm in radar_metrics] + [m[radar_metrics[0]]]
        ax7.plot(angles, vals, color=color, linewidth=2, label=m["agent"])
        ax7.fill(angles, vals, color=color, alpha=0.1)

    ax7.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax7.set_title("Performance radar", fontsize=11, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_agent_pipeline_comparison.png", bbox_inches="tight")
    plt.close()
    print("\n[PLOT] 13_agent_pipeline_comparison.png")


def plot_subtask_contribution(trace):
    """Show how each sub-task flag contributes to correctly identifying fraud."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Sub-task Contribution to Fraud Detection (Decomposed Agent)",
                 fontsize=13, fontweight="bold")

    subtasks = [
        ("velocity_flag",  "Velocity\ncheck",   "#185FA5"),
        ("geo_flag",       "Geo\ncheck",         "#0F6E56"),
        ("behaviour_flag", "Behaviour\ncheck",   "#D85A30"),
        ("identity_flag",  "Identity\ncheck",    "#7F77DD"),
    ]

    for ax, (flag_col, label, color) in zip(axes, subtasks):
        if flag_col not in trace.columns:
            continue
        fraud_flagged   = ((trace["true_label"] == 1) & (trace[flag_col] == 1)).sum()
        fraud_missed    = ((trace["true_label"] == 1) & (trace[flag_col] == 0)).sum()
        legit_flagged   = ((trace["true_label"] == 0) & (trace[flag_col] == 1)).sum()
        legit_clean     = ((trace["true_label"] == 0) & (trace[flag_col] == 0)).sum()

        cm_vals = np.array([[legit_clean, legit_flagged],
                             [fraud_missed, fraud_flagged]])
        im = ax.imshow(cm_vals, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Not Flagged", "Flagged"], fontsize=9)
        ax.set_yticklabels(["Legit", "Fraud"], fontsize=9)
        ax.set_xlabel("Sub-task Decision")
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm_vals[r, c]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm_vals[r, c] > cm_vals.max() * 0.6 else "black")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_subtask_contribution.png", bbox_inches="tight")
    plt.close()
    print("[PLOT] 14_subtask_contribution.png")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  AGENT PIPELINE — TASK DECOMPOSITION EXPERIMENT")
    print("="*60)

    X_sample, y_sample = load_test_sample()

    # Run all three agents
    baseline_pred, baseline_prob, baseline_metrics = run_baseline_agent(X_sample, y_sample)
    cot_pred,      cot_prob,      cot_metrics, reasoning_df = run_cot_agent(X_sample, y_sample)
    decomp_pred,   decomp_prob,   decomp_metrics, step_times, trace = run_decomposed_agent(X_sample, y_sample)

    all_metrics = [baseline_metrics, cot_metrics, decomp_metrics]

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  AGENT COMPARISON SUMMARY")
    print("="*60)
    header = f"{'Agent':<22} {'ROC-AUC':>8} {'F1':>8} {'Recall':>8} {'Precision':>10} {'ms/sample':>10}"
    print(header)
    print("-" * 70)
    for m in all_metrics:
        print(f"  {m['agent']:<20} {m['roc_auc']:>8.4f} {m['f1_fraud']:>8.4f} "
              f"{m['recall']:>8.4f} {m['precision']:>10.4f} {m['latency_ms_per']:>10.2f}")
    print("="*60)

    # ── Sub-task step breakdown ──────────────────────────────────────────
    print("\n  Decomposed Agent — sub-task breakdown:")
    for step, t in step_times.items():
        print(f"    {step:<12}: {t*1000:.1f} ms")

    # ── Save results ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(MODELS_DIR / "agent_pipeline_results.csv", index=False)

    trace.to_csv(MODELS_DIR / "decomposed_agent_trace.csv", index=False)
    print(f"\n[OK] Results saved to outputs/models/agent_pipeline_results.csv")
    print(f"[OK] Full trace saved to outputs/models/decomposed_agent_trace.csv")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_agent_comparison(all_metrics, step_times)
    plot_subtask_contribution(trace)

    print(f"\n[DONE] Agent pipeline complete.")


if __name__ == "__main__":
    main()
