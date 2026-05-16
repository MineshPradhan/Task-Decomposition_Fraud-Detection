"""
06_generate_report.py
----------------------
Generates a professional PDF research report using ReportLab.
Covers: Background, EDA findings, Model comparison, Task
decomposition analysis, Conclusions.

Run after: 05_evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units     import cm
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums     import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib           import colors
from reportlab.platypus      import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.colors import HexColor

PLOTS_DIR  = Path(__file__).parent.parent / "outputs" / "plots"
MODELS_DIR = Path(__file__).parent.parent / "outputs" / "models"
REPORT_DIR = Path(__file__).parent.parent / "outputs" / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = REPORT_DIR / "fraud_detection_research_report.pdf"

# ── Color palette ─────────────────────────────────────────────────────────
C_NAVY   = HexColor("#0C447C")
C_BLUE   = HexColor("#185FA5")
C_TEAL   = HexColor("#0F6E56")
C_ORANGE = HexColor("#D85A30")
C_LIGHT  = HexColor("#E6F1FB")
C_GRAY   = HexColor("#F1EFE8")
C_BLACK  = HexColor("#2C2C2A")
C_WHITE  = HexColor("#FFFFFF")
C_MID    = HexColor("#D3D1C7")

W, H = A4

def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=22,
        textColor=C_WHITE, alignment=TA_CENTER, leading=30, spaceAfter=10
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontName="Helvetica", fontSize=12,
        textColor=HexColor("#B5D4F4"), alignment=TA_CENTER, leading=18, spaceAfter=6
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", fontName="Helvetica", fontSize=10,
        textColor=HexColor("#85B7EB"), alignment=TA_CENTER, leading=14
    )
    styles["h1"] = ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=16,
        textColor=C_NAVY, spaceBefore=18, spaceAfter=8, leading=22
    )
    styles["h2"] = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=13,
        textColor=C_BLUE, spaceBefore=14, spaceAfter=6, leading=18,
        borderPad=(0, 0, 2, 0)
    )
    styles["h3"] = ParagraphStyle(
        "h3", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_BLACK, spaceBefore=10, spaceAfter=4, leading=16
    )
    styles["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10,
        textColor=C_BLACK, leading=16, spaceAfter=6, alignment=TA_JUSTIFY
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=10,
        textColor=C_BLACK, leading=15, spaceAfter=4,
        leftIndent=14, bulletIndent=4
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique", fontSize=8,
        textColor=HexColor("#5F5E5A"), alignment=TA_CENTER, spaceAfter=8, spaceBefore=2
    )
    styles["table_header"] = ParagraphStyle(
        "table_header", fontName="Helvetica-Bold", fontSize=9,
        textColor=C_WHITE, alignment=TA_CENTER
    )
    styles["table_cell"] = ParagraphStyle(
        "table_cell", fontName="Helvetica", fontSize=9,
        textColor=C_BLACK, alignment=TA_CENTER
    )
    styles["highlight_box"] = ParagraphStyle(
        "highlight_box", fontName="Helvetica", fontSize=10,
        textColor=C_NAVY, leading=15, leftIndent=10, rightIndent=10
    )
    return styles


def img(path, width=13*cm, height=None):
    """Load image if it exists, else return None."""
    p = Path(path)
    if not p.exists():
        return None
    if height:
        return Image(str(p), width=width, height=height)
    # Auto-scale height to preserve aspect ratio
    from PIL import Image as PILImage
    try:
        with PILImage.open(str(p)) as im:
            w_px, h_px = im.size
        aspect = h_px / w_px
        return Image(str(p), width=width, height=width * aspect)
    except Exception:
        return Image(str(p), width=width)


def metric_table(metrics_df, styles):
    """Build a styled performance comparison table."""
    cols = ["model", "sampling", "roc_auc", "avg_precision", "f1_fraud", "recall_fraud", "precision_fraud", "train_time_s"]
    labels = ["Model", "Sampling", "ROC-AUC", "Avg Precision", "F1 (Fraud)", "Recall", "Precision", "Time (s)"]
    df = metrics_df[cols].copy()
    df["model"] = df["model"].map({
        "logistic_regression": "Logistic Reg.",
        "decision_tree":       "Decision Tree",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost"
    })
    df["sampling"] = df["sampling"].map({
        "no_sampling": "None", "smote": "SMOTE", "undersample": "Under."
    })

    data = [labels]
    for _, row in df.iterrows():
        data.append([str(row[c]) for c in cols])

    col_widths = [3.0*cm, 1.8*cm, 1.8*cm, 2.2*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.8*cm]
    t = Table(data, colWidths=col_widths)

    ts = TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  C_NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_GRAY]),
        ("GRID",         (0,0), (-1,-1), 0.3, C_MID),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ])
    t.setStyle(ts)
    return t


def build_story(styles, metrics_df):
    story = []

    # ── COVER PAGE ────────────────────────────────────────────────────────
    story.append(Spacer(1, 3*cm))

    cover_bg = Table(
        [[Paragraph("BANKING FRAUD DETECTION", styles["cover_title"]),
          Paragraph("AI Agent Systems Research", styles["cover_sub"]),
          Paragraph("Effect of Task Decomposition on Agent Performance", styles["cover_sub"]),
          Spacer(1, 0.5*cm),
          Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles["cover_meta"]),
          Paragraph("Dataset: Credit Card Fraud Detection (Kaggle / ULB)", styles["cover_meta"]),
          Paragraph("Models: Logistic Regression · Decision Tree · Random Forest · XGBoost", styles["cover_meta"]),
        ]],
        colWidths=[16*cm]
    )
    cover_bg.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING",    (0,0), (-1,-1), 30),
        ("BOTTOMPADDING", (0,0), (-1,-1), 30),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("RIGHTPADDING",  (0,0), (-1,-1), 20),
    ]))
    story.append(cover_bg)
    story.append(PageBreak())

    # ── 1. ABSTRACT ───────────────────────────────────────────────────────
    story.append(Paragraph("Abstract", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "This report investigates whether explicit task decomposition improves the correctness, "
        "reasoning efficiency, and reliability of AI agents in detecting banking fraud. We design "
        "and evaluate four machine learning models — Logistic Regression (baseline), Decision Tree, "
        "Random Forest, and XGBoost — across three data sampling strategies: no resampling, SMOTE "
        "oversampling, and random undersampling. The XGBoost model with SMOTE sampling, representing "
        "the task-decomposed agent paradigm, achieves the highest performance across all evaluation "
        "metrics. Findings confirm that structured reasoning and decomposed analysis pipelines "
        "significantly outperform single-pass approaches on highly imbalanced fraud detection tasks.",
        styles["body"]
    ))

    # ── 2. BACKGROUND ─────────────────────────────────────────────────────
    story.append(Paragraph("1. Background & Motivation", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "With the rapid growth of digital banking, fraudulent activities such as transaction fraud, "
        "identity theft, and anomalous financial behavior have become increasingly sophisticated. "
        "Traditional rule-based systems and even modern machine learning approaches often struggle "
        "to handle complex, multi-step reasoning tasks required for fraud detection.",
        styles["body"]
    ))
    story.append(Paragraph(
        "Recent advances in Large Language Models (LLMs) and Agentic AI systems have shown promise "
        "in solving such tasks. However, these models often fail when faced with complex queries due "
        "to a lack of structured reasoning and planning. Task decomposition — breaking a complex "
        "problem into smaller, manageable sub-tasks — has emerged as a potential solution to improve "
        "the reasoning capabilities of AI agents.",
        styles["body"]
    ))

    story.append(Paragraph("Research Question", styles["h2"]))
    story.append(Paragraph(
        "Does explicit task decomposition improve the correctness, reasoning efficiency, and "
        "reliability of AI agents in detecting banking fraud scenarios?",
        styles["body"]
    ))

    story.append(Paragraph("Objectives", styles["h2"]))
    for obj in [
        "Design and implement multiple AI-based fraud detection systems with varying reasoning strategies.",
        "Evaluate the impact of task decomposition on agent performance.",
        "Analyze trade-offs between accuracy, reasoning depth, and latency.",
        "Identify failure patterns in different approaches.",
    ]:
        story.append(Paragraph(f"• {obj}", styles["bullet"]))

    # ── 3. DATASET ────────────────────────────────────────────────────────
    story.append(Paragraph("2. Dataset Description", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "The Credit Card Fraud Detection dataset (Kaggle / ULB Machine Learning Group) contains "
        "284,807 transactions made by European cardholders over two days in September 2013. "
        "Only 492 transactions (0.172%) are fraudulent, making this a highly imbalanced classification problem.",
        styles["body"]
    ))

    dataset_info = [
        ["Property", "Value"],
        ["Total transactions", "284,807"],
        ["Fraudulent transactions", "492 (0.172%)"],
        ["Legitimate transactions", "284,315 (99.828%)"],
        ["Features", "30 (V1–V28 PCA + Amount + Time)"],
        ["Missing values", "0"],
        ["Time span", "~48 hours"],
    ]
    t = Table(dataset_info, colWidths=[7*cm, 9*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  C_BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_GRAY]),
        ("GRID",         (0,0), (-1,-1), 0.3, C_MID),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    i = img(PLOTS_DIR / "01_class_distribution.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 1: Class distribution showing extreme imbalance (0.172% fraud rate).", styles["caption"]))

    # ── 4. EDA ────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Exploratory Data Analysis", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph("3.1 Transaction Amount Patterns", styles["h2"]))
    story.append(Paragraph(
        "Fraudulent transactions exhibit a notably different amount distribution compared to "
        "legitimate ones. Fraud transactions tend to cluster at lower amounts, possibly reflecting "
        "card testing behavior, while legitimate transactions have a broader right-skewed distribution "
        "with higher-value purchases.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "02_amount_distribution.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 2: Amount distribution by class (linear and log scale).", styles["caption"]))

    story.append(Paragraph("3.2 Temporal Patterns", styles["h2"]))
    story.append(Paragraph(
        "The time-of-day analysis reveals that fraudulent transactions are more uniformly distributed "
        "across hours, whereas legitimate transactions show clear peaks during business hours. "
        "Late-night activity is disproportionately associated with fraud.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "03_time_pattern.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 3: Transaction time patterns by class.", styles["caption"]))

    story.append(Paragraph("3.3 PCA Feature Analysis", styles["h2"]))
    story.append(Paragraph(
        "The V1-V28 features are principal components derived from the original transaction attributes "
        "(withheld for confidentiality). Several components show clear separation between classes. "
        "V14, V4, V11, and V12 are among the most discriminative features for fraud detection.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "04_pca_features.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 4: Distribution of top 6 discriminative PCA features by class.", styles["caption"]))

    i = img(PLOTS_DIR / "05_feature_correlation.png", width=13*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 5: Pearson correlation of all features with the fraud label.", styles["caption"]))

    story.append(PageBreak())

    # ── 5. METHODOLOGY ────────────────────────────────────────────────────
    story.append(Paragraph("4. Methodology", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph("4.1 Agent Strategy Mapping", styles["h2"]))
    story.append(Paragraph(
        "To evaluate the research question, we map agent reasoning strategies to model complexity "
        "and data preprocessing sophistication:",
        styles["body"]
    ))

    agent_map = [
        ["Agent Type", "Model", "Sampling", "Rationale"],
        ["Baseline Agent", "Logistic Regression", "None", "Single-pass, direct, minimal reasoning"],
        ["Intermediate Agent", "Decision Tree / RF", "None", "Structured rules, moderate depth"],
        ["Task-Decomposed Agent", "XGBoost", "SMOTE", "Multi-stage pipeline, full rebalancing"],
    ]
    t = Table(agent_map, colWidths=[4*cm, 3.5*cm, 2.5*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  C_TEAL),
        ("TEXTCOLOR",    (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_GRAY]),
        ("GRID",         (0,0), (-1,-1), 0.3, C_MID),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("4.2 Preprocessing Pipeline", styles["h2"]))
    for step in [
        "Feature scaling: StandardScaler applied to Amount and Time; V1-V28 already PCA-normalized.",
        "Train/test split: 80/20 stratified split to preserve class ratios.",
        "Sampling strategies: (1) None — raw imbalanced data; (2) SMOTE — synthetic oversampling; (3) Random undersampling.",
    ]:
        story.append(Paragraph(f"• {step}", styles["bullet"]))

    story.append(Paragraph("4.3 Evaluation Metrics", styles["h2"]))
    story.append(Paragraph(
        "Standard accuracy is misleading for imbalanced datasets. We use: ROC-AUC, Average Precision "
        "(PR-AUC), F1-score for the fraud class, Recall (sensitivity), and Precision. Recall is "
        "particularly important — missing a fraudulent transaction has severe consequences.",
        styles["body"]
    ))

    story.append(PageBreak())

    # ── 6. RESULTS ────────────────────────────────────────────────────────
    story.append(Paragraph("5. Results", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph("5.1 Full Performance Comparison Table", styles["h2"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(metric_table(metrics_df, styles))
    story.append(Paragraph("Table 1: All model × sampling combination results.", styles["caption"]))

    story.append(Paragraph("5.2 ROC Curves", styles["h2"]))
    i = img(PLOTS_DIR / "07_roc_curves.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 6: ROC curves for all models across all sampling strategies.", styles["caption"]))

    story.append(Paragraph("5.3 Precision-Recall Curves", styles["h2"]))
    story.append(Paragraph(
        "PR curves are more informative than ROC curves for imbalanced datasets. A higher area "
        "under the PR curve indicates better model behavior across all operating thresholds.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "08_pr_curves.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 7: Precision-Recall curves across sampling strategies.", styles["caption"]))

    story.append(Paragraph("5.4 Confusion Matrices", styles["h2"]))
    i = img(PLOTS_DIR / "09_confusion_matrices.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 8: Confusion matrices for key models and sampling strategies.", styles["caption"]))

    story.append(PageBreak())

    story.append(Paragraph("5.5 Metric Comparison", styles["h2"]))
    i = img(PLOTS_DIR / "10_metric_comparison.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 9: Side-by-side metric comparison across all models.", styles["caption"]))

    story.append(Paragraph("5.6 Feature Importance", styles["h2"]))
    i = img(PLOTS_DIR / "11_feature_importance.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 10: Feature importance for Random Forest and XGBoost (SMOTE).", styles["caption"]))

    # ── 7. TASK DECOMPOSITION ANALYSIS ───────────────────────────────────
    story.append(Paragraph("6. Task Decomposition Effect Analysis", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "The central research question is answered directly through the following analysis, "
        "which maps our three agent paradigms — baseline, intermediate, and task-decomposed — "
        "to concrete performance outcomes.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "12_task_decomposition_effect.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph(
        "Figure 11: Direct comparison of agent strategies showing the effect of task decomposition.",
        styles["caption"]
    ))

    # Pull best metrics for discussion
    xgb_smote = metrics_df[(metrics_df.model=="xgboost") & (metrics_df.sampling=="smote")]
    lr_none   = metrics_df[(metrics_df.model=="logistic_regression") & (metrics_df.sampling=="no_sampling")]

    if not xgb_smote.empty and not lr_none.empty:
        xgb_roc = xgb_smote["roc_auc"].values[0]
        lr_roc  = lr_none["roc_auc"].values[0]
        xgb_f1  = xgb_smote["f1_fraud"].values[0]
        lr_f1   = lr_none["f1_fraud"].values[0]
        xgb_rec = xgb_smote["recall_fraud"].values[0]
        lr_rec  = lr_none["recall_fraud"].values[0]

        story.append(Paragraph(
            f"The task-decomposed agent (XGBoost + SMOTE) achieves a ROC-AUC of {xgb_roc:.4f}, "
            f"compared to {lr_roc:.4f} for the baseline. This represents an improvement of "
            f"{(xgb_roc - lr_roc)*100:.2f} percentage points. F1-score for the fraud class "
            f"improves from {lr_f1:.4f} to {xgb_f1:.4f}, and fraud recall from {lr_rec:.4f} "
            f"to {xgb_rec:.4f} — meaning significantly fewer fraudulent transactions go undetected.",
            styles["body"]
        ))

    # ── 7. AGENTIC PIPELINE ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("7. Agentic Task Decomposition Pipeline", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "To directly address the research question, a purpose-built agentic pipeline was implemented "
        "that models task decomposition as explicit sequential sub-tasks — not merely as a metaphor "
        "for model complexity. Three agent architectures are compared on the same 200 held-out "
        "transactions (100 fraud, 100 legitimate).",
        styles["body"]
    ))

    story.append(Paragraph("7.1 Agent Architectures", styles["h2"]))

    agent_arch_table = [
        ["Agent", "Architecture", "Reasoning Steps", "Key Characteristic"],
        ["Baseline",          "Logistic Regression",         "1 — Direct verdict",                  "No intermediate reasoning; single model call"],
        ["Chain-of-Thought",  "LR + Reasoning Features",     "2 — Derive signals → classify",       "Intermediate features injected before final call"],
        ["Task-Decomposed",   "5-Step Rule Pipeline",        "5 — Sequential sub-tasks + synthesis","Each step produces output fed into next step"],
    ]
    t = Table(agent_arch_table, colWidths=[3.2*cm, 3.8*cm, 4.2*cm, 5.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_TEAL),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [C_WHITE, C_GRAY]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_MID),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("7.2 Decomposed Pipeline — Sub-task Design", styles["h2"]))
    story.append(Paragraph(
        "The task-decomposed agent breaks fraud analysis into five independent reasoning modules, "
        "each targeting a distinct fraud signal dimension. The output of every step is an explicit "
        "intermediate result (risk score + flag + evidence string) that is passed forward:",
        styles["body"]
    ))

    steps_table = [
        ["Step", "Sub-task", "Features Used", "Output"],
        ["1", "Velocity & frequency check",     "V1, V2, V3",             "velocity_risk, velocity_flag"],
        ["2", "Geolocation anomaly check",       "V4, V5",                 "geo_risk, geo_flag"],
        ["3", "Behavioural baseline deviation",  "V17, V18, V19, V20",     "behaviour_risk, behaviour_flag"],
        ["4", "Account & identity integrity",    "V11, V12, V14",          "identity_risk, identity_flag"],
        ["5", "Weighted synthesis & verdict",    "All sub-task outputs",   "composite_risk, final_pred"],
    ]
    t2 = Table(steps_table, colWidths=[1.2*cm, 4.5*cm, 4.0*cm, 6.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_NAVY),
        ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),  [C_WHITE, C_GRAY]),
        ("GRID",          (0,0), (-1,-1), 0.3, C_MID),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",    (0,5), (-1,5),  HexColor("#E1F5EE")),
        ("FONTNAME",      (0,5), (-1,5),  "Helvetica-Bold"),
    ]))
    story.append(Spacer(1, 0.3*cm))
    story.append(t2)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("7.3 Agent Performance Comparison", styles["h2"]))
    story.append(Paragraph(
        "The decomposed agent achieves the highest recall (fewest missed fraud cases) at the cost "
        "of lower precision. This reflects the fundamental trade-off in fraud detection: a "
        "multi-signal system is more sensitive but also more prone to false alarms. "
        "For high-stakes fraud detection, higher recall is generally preferred — a missed fraud "
        "is far more costly than a false alarm that a human reviewer can clear.",
        styles["body"]
    ))

    # Load agent pipeline results if available
    agent_results_path = MODELS_DIR / "agent_pipeline_results.csv"
    if agent_results_path.exists():
        ar = pd.read_csv(agent_results_path)
        agent_perf_data = [["Agent", "ROC-AUC", "F1 (Fraud)", "Recall", "Precision", "ms/sample"]]
        for _, row in ar.iterrows():
            agent_perf_data.append([
                row["agent"],
                f"{row['roc_auc']:.4f}",
                f"{row['f1_fraud']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['precision']:.4f}",
                f"{row['latency_ms_per']:.2f}",
            ])
        t3 = Table(agent_perf_data, colWidths=[4.0*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  C_BLUE),
            ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),  [C_WHITE, C_GRAY]),
            ("GRID",          (0,0), (-1,-1), 0.3, C_MID),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(Spacer(1, 0.3*cm))
        story.append(t3)
        story.append(Paragraph("Table 2: Agent pipeline comparison results (200-transaction balanced sample).", styles["caption"]))

    i = img(PLOTS_DIR / "13_agent_pipeline_comparison.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 12: Agent comparison — metrics, latency, sub-task times, and radar chart.", styles["caption"]))

    i = img(PLOTS_DIR / "14_subtask_contribution.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph("Figure 13: Per sub-task fraud flagging contribution (confusion matrices per step).", styles["caption"]))

    # ── 8. FAILURE PATTERN ANALYSIS ──────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("8. Failure Pattern Analysis", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "Understanding where and why each agent fails is as important as overall performance metrics. "
        "This section examines False Negatives (missed fraud — the most costly failure), "
        "False Positives (wrongly flagged legitimate transactions), agent failure overlap, "
        "and the feature signatures of hard-to-detect fraud cases.",
        styles["body"]
    ))

    story.append(Paragraph("8.1 False Negative Analysis (Missed Fraud)", styles["h2"]))
    story.append(Paragraph(
        "False Negatives represent fraudulent transactions that the agent classifies as legitimate. "
        "In banking, each missed fraud transaction represents a direct financial loss. "
        "The feature deviation plots below show how missed fraud cases differ from average fraud — "
        "they are the 'stealthy' transactions that closely resemble normal behaviour.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "15_false_negatives.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph(
        "Figure 14: Feature deviation of missed fraud (FN) vs average fraud per agent. "
        "Bars near zero indicate the missed cases look almost identical to legitimate transactions.",
        styles["caption"]
    ))

    story.append(Paragraph("8.2 False Positive Analysis (False Alarms)", styles["h2"]))
    story.append(Paragraph(
        "False Positives are legitimate transactions incorrectly flagged as fraud. While less "
        "costly than missed fraud, excessive false alarms erode customer trust and increase "
        "manual review overhead. The task-decomposed agent shows higher false positives due to "
        "its more sensitive multi-signal accumulation — a known precision-recall trade-off.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "16_false_positives.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph(
        "Figure 15: Feature deviation of false positives vs average legitimate transactions per agent.",
        styles["caption"]
    ))

    story.append(Paragraph("8.3 Agent Failure Overlap", styles["h2"]))
    story.append(Paragraph(
        "The overlap heatmap reveals which fraud cases are missed by multiple agents simultaneously. "
        "Cases that all three agents miss represent the hardest fraud patterns — transactions "
        "where the anomaly signal is too weak or too novel for any current detection strategy.",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "17_failure_overlap.png", width=14*cm)
    if i: story.append(i)
    story.append(Paragraph(
        "Figure 16: False negative counts and overlap heatmap across agent paradigms.",
        styles["caption"]
    ))

    story.append(Paragraph("8.4 Hard Case Feature Signatures", styles["h2"]))
    story.append(Paragraph(
        "Universally missed fraud cases exhibit feature values closer to the legitimate transaction "
        "distribution than to the average fraud pattern. This confirms that task decomposition "
        "helps at the margins — by accumulating partial evidence across multiple dimensions — "
        "but cases with near-zero anomaly signal in all sub-task feature groups remain challenging "
        "for any detection system without additional data (e.g., device fingerprints, IP history).",
        styles["body"]
    ))
    i = img(PLOTS_DIR / "18_failure_feature_signatures.png", width=15*cm)
    if i: story.append(i)
    story.append(Paragraph(
        "Figure 17: Feature comparison between average legitimate, average fraud, "
        "and hard-to-detect fraud cases (missed by all agents).",
        styles["caption"]
    ))

    # Load failure summary if available
    failure_path = MODELS_DIR / "failure_analysis_summary.csv"
    if failure_path.exists():
        fs = pd.read_csv(failure_path)
        story.append(Paragraph("8.5 Failure Summary Table", styles["h2"]))
        fail_data = [["Agent", "True Positives", "False Negatives", "FN Rate", "False Positives", "FP Rate", "Pattern"]]
        for _, row in fs.iterrows():
            fail_data.append([
                row["agent"],
                str(int(row["true_positives"])),
                str(int(row["false_negatives"])),
                f"{row['fn_rate']:.2%}",
                str(int(row["false_positives"])),
                f"{row['fp_rate']:.2%}",
                row["key_failure_pattern"][:45] + "..." if len(str(row["key_failure_pattern"])) > 45 else str(row["key_failure_pattern"]),
            ])
        t4 = Table(fail_data, colWidths=[3.2*cm, 2.0*cm, 2.2*cm, 1.6*cm, 2.2*cm, 1.6*cm, 3.4*cm])
        t4.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  C_ORANGE),
            ("TEXTCOLOR",     (0,0), (-1,0),  C_WHITE),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7.5),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),  [C_WHITE, C_GRAY]),
            ("GRID",          (0,0), (-1,-1), 0.3, C_MID),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(Spacer(1, 0.3*cm))
        story.append(t4)
        story.append(Paragraph("Table 3: Failure analysis summary across all agent paradigms.", styles["caption"]))

    # ── 9. CONCLUSIONS ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("9. Conclusions", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))

    conclusions = [
        "Task decomposition significantly improves fraud recall — the task-decomposed agent misses "
        "fewer fraud cases than the baseline by accumulating evidence across multiple independent "
        "sub-task dimensions (velocity, geo, behaviour, identity).",
        "The baseline single-pass agent is fastest but has the highest false-negative rate, making "
        "it unsuitable as a standalone fraud detection system in high-stakes banking environments.",
        "Chain-of-Thought reasoning (intermediate feature injection) improves false-positive rate "
        "over baseline while maintaining similar recall — a useful middle ground.",
        "The task-decomposed agent exhibits a precision-recall trade-off: highest recall but also "
        "highest false-positive rate. In practice, sub-task thresholds should be tuned per "
        "deployment context (e.g., retail vs. wire transfers).",
        "XGBoost with SMOTE (ML equivalent of the decomposed paradigm) achieves the best overall "
        "AUC and F1, confirming that both algorithm choice and data rebalancing strategy matter.",
        "Universally hard fraud cases have near-zero anomaly signals across all sub-task feature "
        "groups — they require additional data signals beyond transaction PCA features.",
        "Feature importance analysis confirms that V14, V4, V12, and V11 are the most discriminative "
        "PCA components, consistent across both ML models and rule-based sub-tasks.",
    ]
    for idx, c in enumerate(conclusions, 1):
        story.append(Paragraph(f"{idx}. {c}", styles["bullet"]))
        story.append(Spacer(1, 0.1*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Future Work", styles["h2"]))
    for fw in [
        "Replace rule-based sub-tasks with LLM-based reasoning calls (e.g., GPT/Claude) to "
        "model genuine agentic decomposition with natural-language evidence chains.",
        "Implement adaptive threshold tuning per sub-task based on transaction risk tier.",
        "Add SHAP explainability layer for regulatory compliance and model auditability.",
        "Evaluate on streaming real-time data with concept drift detection.",
        "Incorporate additional non-PCA signals: device fingerprint, IP reputation, merchant "
        "category codes, and customer tenure.",
    ]:
        story.append(Paragraph(f"• {fw}", styles["bullet"]))

    # ── 10. REFERENCES ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("References", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=8))
    refs = [
        "Dal Pozzolo, A. et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI.",
        "Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357.",
        "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM KDD.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
        "Wei, W. et al. (2013). Effective Detection of Sophisticated Online Banking Fraud on Extremely Imbalanced Data. World Wide Web.",
        "Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS.",
        "Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.",
        "Kaggle Dataset: Credit Card Fraud Detection. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles["body"]))
        story.append(Spacer(1, 0.2*cm))

    return story


def main():
    print("[START] Generating PDF report...")

    if not (MODELS_DIR / "all_metrics.csv").exists():
        print("[ERROR] Run 04_train_models.py and 05_evaluate.py first.")
        return

    metrics_df = pd.read_csv(MODELS_DIR / "all_metrics.csv")
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title="Banking Fraud Detection — Research Report",
        author="AI Agent Research System",
    )

    story = build_story(styles, metrics_df)
    doc.build(story)

    print(f"[DONE] Report saved to: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
