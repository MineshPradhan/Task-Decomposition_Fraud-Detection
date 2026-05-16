# 🛡️ Effect of Task Decomposition on Agent Performance in Banking Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE Format](https://img.shields.io/badge/Paper-IEEE%20Format-blue)](outputs/report/)

> **Research Question:** Does explicit task decomposition improve the correctness, reasoning efficiency, and reliability of AI agents in detecting banking fraud scenarios?

---

## 📌 Overview

This project implements and compares **three AI agent reasoning paradigms** for banking fraud detection, directly investigating the effect of task decomposition on agent performance. It combines a comprehensive classical ML benchmark (12 model–sampling combinations) with an explicit agentic pipeline and a dedicated failure pattern analysis.

The project is structured as a reproducible research pipeline with a full IEEE-format paper, 18 evaluation plots, and per-transaction reasoning traces.

---

## 🔬 Research Design

| Agent | Reasoning Strategy | Architecture | Key Idea |
|---|---|---|---|
| **Baseline** | Single-pass | Logistic Regression | One model call, no intermediate steps |
| **Chain-of-Thought** | Reasoning-augmented | LR + 5 derived signals | Intermediate features injected before classification |
| **Task-Decomposed** | Sequential pipeline | 5-step rule chain | Each step produces output fed into the next |

The Task-Decomposed agent runs five sequential sub-tasks:

```
Velocity Check → Geo Anomaly → Behavioural Deviation → Identity Integrity → Synthesis
     (V1–V3)         (V4–V5)          (V17–V20)            (V11,V12,V14)
```

---

## 📊 Key Results

### Agent Pipeline Comparison

| Agent | ROC-AUC | F1 (Fraud) | Recall | Precision | ms/sample |
|---|---|---|---|---|---|
| Baseline | **0.9994** | **0.9794** | 0.9694 | **0.9896** | 0.02 |
| Chain-of-Thought | **0.9994** | **0.9794** | 0.9694 | **0.9896** | 147.51 |
| **Task-Decomposed** | 0.9976 | 0.9282 | **0.9898** | 0.8739 | **0.26** |

> ✅ Task-Decomposed agent achieves **98.98% recall** — catching 97/98 fraud cases vs. 95/98 for Baseline.

### Best ML Model Results (XGBoost + SMOTE)

| Metric | Score |
|---|---|
| ROC-AUC | **0.9986** |
| Average Precision | **0.9336** |
| F1 (Fraud) | 0.6039 |
| Recall | 0.9490 |
| Precision | 0.4429 |

### Failure Analysis Summary

| Agent | True Positives | False Negatives | FN Rate | FP Rate |
|---|---|---|---|---|
| Baseline | 95 | 3 | 3.06% | 0.35% |
| Chain-of-Thought | 95 | 3 | 3.06% | 0.30% |
| **Task-Decomposed** | **97** | **1** | **1.02%** | 20.04% |

---

## 📁 Project Structure

```
fraud_detection/
│
├── run_pipeline.py              # 🚀 Master script — runs full pipeline end-to-end
├── requirements.txt             # Python dependencies
├── README.md
│
├── scripts/
│   ├── 01_download_data.py      # Download Kaggle dataset or generate synthetic data
│   ├── 02_eda.py                # Exploratory Data Analysis (6 plots)
│   ├── 03_preprocess.py         # Feature scaling + 3 sampling strategies
│   ├── 04_train_models.py       # Train 4 models × 3 sampling = 12 experiments
│   ├── 05_evaluate.py           # ROC, PR curves, confusion matrices, comparisons
│   ├── 06_generate_report.py    # PDF research report (ReportLab)
│   ├── 07_agent_pipeline.py     # Baseline vs CoT vs Task-Decomposed agent
│   └── 08_failure_analysis.py   # FN/FP profiles, overlap, hard-case signatures
│
├── data/
│   ├── creditcard.csv           # Dataset (downloaded or synthetic)
│   └── processed/               # Train/test splits per sampling strategy
│
└── outputs/
    ├── plots/                   # 18 generated PNG plots
    ├── models/                  # Trained models (.pkl) + metrics CSVs
    └── report/                  # PDF research report
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-task-decomposition.git
cd fraud-detection-task-decomposition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the dataset

**Option A — Kaggle API (recommended, gets real data):**
```bash
# Place kaggle.json in ~/.kaggle/
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
Download from: [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
Place `creditcard.csv` in `data/`

**Option B — Auto-generate synthetic data (no Kaggle needed):**  
The pipeline detects a missing dataset and automatically generates a synthetic equivalent with identical schema and fraud ratio (0.172%).

### 4. Run the full pipeline

```bash
python run_pipeline.py
```

Or run individual steps:

```bash
python scripts/01_download_data.py    # Step 1: Data
python scripts/02_eda.py              # Step 2: EDA
python scripts/03_preprocess.py       # Step 3: Preprocessing
python scripts/04_train_models.py     # Step 4: Train models
python scripts/05_evaluate.py         # Step 5: Evaluation plots
python scripts/07_agent_pipeline.py   # Step 6: Agent comparison
python scripts/08_failure_analysis.py # Step 7: Failure analysis
python scripts/06_generate_report.py  # Step 8: PDF report
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
reportlab>=3.6.0
Pillow>=9.0.0
kaggle>=1.5.0       # optional, for real dataset download
```

> ✅ **Recommended Python version: 3.10**

---

## 🧪 Dataset

| Property | Value |
|---|---|
| Source | [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraud cases | 492 (0.172%) |
| Features | 30 (V1–V28 PCA + Amount + Time) |
| Missing values | 0 |
| Time span | 48 hours |

Features V1–V28 are PCA-transformed (original features anonymised). Amount and Time are the only raw features.

---

## 🤖 Agent Architecture Detail

### Agent 3 — Task-Decomposed Pipeline

```python
# Step 1: Velocity check (V1, V2, V3)
vel_risk  = clip(mean(|V1|,|V2|,|V3|) / 3.0, 0, 1)
vel_flag  = vel_risk > 0.4

# Step 2: Geolocation anomaly (V4, V5)
geo_risk  = clip(mean(|V4|,|V5|) / 3.5, 0, 1)
geo_flag  = geo_risk > 0.35

# Step 3: Behavioural deviation (V17–V20)
beh_risk  = clip(mean(|V17|...|V20|) / 3.0, 0, 1)
beh_flag  = beh_risk > 0.4

# Step 4: Identity integrity (V11, V12, V14)
id_risk   = clip(mean(|V11|,|V12|,|V14|) / 4.0, 0, 1)
id_flag   = id_risk > 0.35

# Step 5: Weighted synthesis
composite = (2*vel + 1*geo + 2*beh + 3*id) / 8
fraud     = (composite > 0.28) OR (flags_raised >= 3)
```

Each step produces an intermediate output (risk score + flag + evidence string) stored in `outputs/models/decomposed_agent_trace.csv`.

---

## 📈 Output Files

After running the full pipeline:

| Output | Location | Description |
|---|---|---|
| EDA plots | `outputs/plots/01–06_*.png` | Class distribution, amounts, time, PCA features, correlations |
| Evaluation plots | `outputs/plots/07–12_*.png` | ROC, PR curves, confusion matrices, metric comparison |
| Agent comparison | `outputs/plots/13–14_*.png` | Agent pipeline comparison, sub-task contribution |
| Failure analysis | `outputs/plots/15–18_*.png` | FN profiles, FP profiles, overlap heatmap, hard-case signatures |
| ML metrics | `outputs/models/all_metrics.csv` | All 12 model–sampling results |
| Agent results | `outputs/models/agent_pipeline_results.csv` | Baseline vs CoT vs Decomposed |
| Agent trace | `outputs/models/decomposed_agent_trace.csv` | Per-transaction sub-task reasoning trace |
| Failure summary | `outputs/models/failure_analysis_summary.csv` | FN/FP counts and patterns per agent |
| Research report | `outputs/report/fraud_detection_research_report.pdf` | Full PDF report |


## 🧩 Methodology Summary

```
Dataset (284,807 transactions)
        │
        ├── EDA (6 plots)
        │
        ├── Preprocessing
        │   ├── StandardScaler (Amount, Time)
        │   ├── Stratified 80/20 split
        │   └── 3 sampling strategies: None / SMOTE / Undersample
        │
        ├── ML Benchmark (12 experiments)
        │   ├── Logistic Regression
        │   ├── Decision Tree
        │   ├── Random Forest
        │   └── XGBoost
        │
        ├── Agent Pipeline (3 paradigms)
        │   ├── Baseline        → single-pass LR
        │   ├── Chain-of-Thought → LR + 5 reasoning features
        │   └── Task-Decomposed → 5-step sequential pipeline
        │
        ├── Failure Analysis
        │   ├── False Negative profiles per agent
        │   ├── False Positive profiles per agent
        │   ├── Agent failure overlap heatmap
        │   └── Hard-case feature signatures
        │
        └── PDF Research Report (IEEE format)
```

---

## 📌 Findings

1. **Task decomposition improves recall** — the Task-Decomposed agent reduces missed fraud from 3 to 1 case (FN rate: 3.06% → 1.02%).
2. **Precision-recall trade-off** — higher recall comes with elevated false positives (0.35% → 20.04%); threshold tuning is recommended for production.
3. **XGBoost + SMOTE** is the best single ML model (ROC-AUC 0.9986, AP 0.9336).
4. **CoT agent matches Baseline** when the same classifier is used, since derived features add no new information for linear models.
5. **One universally hard fraud case** exists whose PCA feature profile is indistinguishable from legitimate transactions — requires non-PCA signals to resolve.

---

## 🔮 Future Work

- Replace statistical sub-tasks with **LLM API reasoning calls** for genuine agentic decomposition
- **Adaptive threshold calibration** per sub-task using institution-specific cost functions
- **SHAP explainability** layer for regulatory compliance
- **Streaming real-time evaluation** with concept drift detection
- Incorporate additional signals: device fingerprint, IP reputation, merchant category codes

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [ULB Machine Learning Group](https://mlg.ulb.ac.be/) for the Credit Card Fraud Detection dataset
- [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) for hosting the dataset
- Open-source libraries: scikit-learn, XGBoost, imbalanced-learn, ReportLab, matplotlib

---

<p align="center">
  Made for research purposes · IEEE-format paper included · Python 3.10
</p>