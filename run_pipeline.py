"""
run_pipeline.py
----------------
Master script — runs the full pipeline end-to-end:
  1. Download / generate dataset
  2. Exploratory Data Analysis
  3. Preprocessing & sampling
  4. Train all models
  5. Evaluate & generate plots
  6. Generate PDF research report

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"

STEPS = [
    ("01_download_data.py",   "Data Download / Generation"),
    ("02_eda.py",             "Exploratory Data Analysis"),
    ("03_preprocess.py",      "Preprocessing & Sampling"),
    ("04_train_models.py",    "Model Training"),
    ("05_evaluate.py",        "Evaluation & Plots"),
    ("07_agent_pipeline.py",  "Agentic Task Decomposition Pipeline"),
    ("08_failure_analysis.py","Failure Pattern Analysis"),
    ("06_generate_report.py", "PDF Report Generation"),
]


def run_step(script, label):
    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  Script: {script}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script)],
        capture_output=False
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {label}")
        sys.exit(1)
    print(f"\n[OK] Completed in {elapsed:.1f}s")
    return elapsed


def main():
    print("\n" + "="*60)
    print("  BANKING FRAUD DETECTION — FULL PIPELINE")
    print("  Effect of Task Decomposition on Agent Performance")
    print("="*60)

    total_start = time.time()
    timings = {}

    for script, label in STEPS:
        timings[label] = run_step(script, label)

    total = time.time() - total_start

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE — TIMING SUMMARY")
    print("="*60)
    for label, t in timings.items():
        print(f"  {label:<35} {t:>6.1f}s")
    print(f"  {'TOTAL':<35} {total:>6.1f}s")
    print("="*60)
    print("\n  Outputs:")
    print("  - EDA plots         : outputs/plots/01_*.png ... 06_*.png")
    print("  - Eval plots        : outputs/plots/07_*.png ... 12_*.png")
    print("  - Agent pipeline    : outputs/plots/13_agent_pipeline_comparison.png")
    print("  -                     outputs/plots/14_subtask_contribution.png")
    print("  - Failure analysis  : outputs/plots/15_*.png ... 18_*.png")
    print("  - Trained models    : outputs/models/*.pkl")
    print("  - Metrics CSV       : outputs/models/all_metrics.csv")
    print("  - Agent results     : outputs/models/agent_pipeline_results.csv")
    print("  - Agent trace       : outputs/models/decomposed_agent_trace.csv")
    print("  - Failure summary   : outputs/models/failure_analysis_summary.csv")
    print("  - Research report   : outputs/report/fraud_detection_research_report.pdf")
    print()


if __name__ == "__main__":
    main()
