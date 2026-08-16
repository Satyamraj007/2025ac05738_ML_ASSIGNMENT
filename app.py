import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: #f8f9ff;
        border-left: 4px solid #0f3460;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
    }
    .winner-badge {
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .section-header {
        background: #f0f4ff;
        border-left: 4px solid #0f3460;
        padding: 0.5rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0 0.8rem 0;
        font-weight: bold;
        color: #1a1a2e;
    }
    div[data-testid="stMetric"] {
        background: #f8f9ff;
        border: 1px solid #e0e6ff;
        border-radius: 10px;
        padding: 0.8rem;
    }
    div[data-testid="stMetric"] label { color: #555 !important; font-size: 0.8rem !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #0f3460 !important; font-size: 1.5rem !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

BOOL_COLS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"
]
CAT_COLS     = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
THRESH       = 0.15
RANDOM_STATE = 42
DATA_PATH    = "test_data.csv"

BEST_MODEL = "Random Forest"

MODEL_INFO = {
    "Logistic Regression": "Linear model. Excels at recall — catches almost all fraud but produces many false alarms.",
    "Decision Tree":       "Rule-based tree. Interpretable but weakest overall discriminator on this dataset.",
    "kNN":                 "Distance-based. High accuracy but poor recall on rare fraud class.",
    "Naive Bayes":         "Probabilistic. Strong AUC and recall among single models. Fast and effective.",
    "Random Forest":       "Ensemble of 300 trees. Best overall — highest F1, MCC and AUC.",
}


def make_preprocessor(kind, numeric_cols):
    if kind == "scaled":
        return ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ])
    return ColumnTransformer([
        ("num", "passthrough", numeric_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
    ])


def preprocess_df(df):
    df = df.copy()
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])
    for c in BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower().map(
                {"true": 1, "false": 0, "1": 1, "0": 0}
            ).fillna(0).astype(int)
    return df


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1 Score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


@st.cache_resource
def train_all_models(data_key=None):
    if data_key and "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"].copy()
    else:
        df = pd.read_csv(DATA_PATH)
    df = preprocess_df(df)

    numeric_cols = [c for c in df.columns if c not in CAT_COLS + ["is_fraud"]]
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = {
        "Logistic Regression": Pipeline([("pre", make_preprocessor("scaled", numeric_cols)), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Decision Tree":       Pipeline([("pre", make_preprocessor("trees",  numeric_cols)), ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE))]),
        "kNN":                 Pipeline([("pre", make_preprocessor("scaled", numeric_cols)), ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        "Naive Bayes":         Pipeline([("pre", make_preprocessor("scaled", numeric_cols)), ("clf", GaussianNB(var_smoothing=1e-8))]),
        "Random Forest":       Pipeline([("pre", make_preprocessor("trees",  numeric_cols)), ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", max_depth=12, random_state=RANDOM_STATE, n_jobs=-1))]),
    }

    trained = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        trained[name] = pipe

    return trained, X_test, y_test, numeric_cols


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Credit Card Fraud Detection</h1>
    <p>Binary Classification using 5 ML Models &nbsp;|&nbsp; 20,000 transactions &nbsp;|&nbsp; 25 features &nbsp;|&nbsp; Threshold = 0.15</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")
model_name = st.sidebar.selectbox(
    "Select ML Model",
    list(MODEL_INFO.keys()),
    format_func=lambda x: ("★ " if x == BEST_MODEL else "  ") + x
)
st.sidebar.markdown(f"*{MODEL_INFO[model_name]}*")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.markdown("- 20,000 transactions\n- 25 features\n- 1.7% fraud rate\n- 80/20 train/test split")
st.sidebar.markdown("---")
st.sidebar.caption("BITS Pilani | M.Tech AIML\nStudent: 2025ac05738")

# ── Dataset Upload ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">1. Dataset Upload (CSV)</div>', unsafe_allow_html=True)

import os
if st.session_state.get("uploaded_file_id"):
    fid = st.session_state["uploaded_file_id"]
    st.success(f"Active dataset: {fid[0]} — {fid[1] // 1024} KB")
else:
    size_kb = os.path.getsize(DATA_PATH) // 1024
    st.success(f"Active dataset: test_data.csv — {size_kb} KB (20,000 transactions, default from repo)")

uploaded = st.file_uploader("Upload a different CSV to retrain on new data", type=["csv"])

if uploaded:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("uploaded_file_id") != file_id:
        try:
            df_check = pd.read_csv(uploaded)
        except Exception:
            st.error("Corrupted file. Please upload a valid CSV file.")
            st.stop()
        if "is_fraud" not in df_check.columns:
            st.error("Column 'is_fraud' not found. Please upload the correct CSV.")
            st.stop()
        st.session_state["uploaded_df"] = df_check
        st.session_state["uploaded_file_id"] = file_id
        st.cache_resource.clear()
        st.rerun()

# ── Train ─────────────────────────────────────────────────────────────────────
with st.spinner("Training all 5 models on 80% data... this may take a moment."):
    trained_models, X_test, y_test, numeric_cols = train_all_models(
        data_key=st.session_state.get("uploaded_file_id")
    )

st.success(f"All 5 models trained. Evaluating on held-out 20% test split ({len(y_test):,} rows).")

pipeline = trained_models[model_name]
y_true   = y_test.values
y_prob   = pipeline.predict_proba(X_test)[:, 1]
y_pred   = (y_prob >= THRESH).astype(int)

# ── Metrics ───────────────────────────────────────────────────────────────────
badge = '<span class="winner-badge">BEST MODEL</span>' if model_name == BEST_MODEL else ""
st.markdown(f'<div class="section-header">2. Evaluation Metrics — {model_name} {badge}</div>', unsafe_allow_html=True)

metrics = compute_metrics(y_true, y_pred, y_prob)
cols = st.columns(len(metrics))
metric_colors = {"Accuracy": "normal", "AUC": "normal", "Precision": "normal",
                 "Recall": "normal", "F1 Score": "normal", "MCC": "normal"}
for i, (k, v) in enumerate(metrics.items()):
    cols[i].metric(k, v)

# ── All Models Comparison ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">3. All Models Comparison</div>', unsafe_allow_html=True)
comparison_rows = []
for mn, pipe in trained_models.items():
    yp   = pipe.predict_proba(X_test)[:, 1]
    yprd = (yp >= THRESH).astype(int)
    row  = {"Model": mn}
    row.update(compute_metrics(y_true, yprd, yp))
    comparison_rows.append(row)

df_cmp = pd.DataFrame(comparison_rows).set_index("Model")
st.dataframe(
    df_cmp.style
    .highlight_max(axis=0, color="#c8f7c5")
    .highlight_min(axis=0, color="#fde8e8")
    .format("{:.4f}"),
    use_container_width=True
)
st.caption("Green = best per metric   |   Red = worst per metric   |   Threshold = 0.15 for all models")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
st.markdown(f'<div class="section-header">4. Confusion Matrix — {model_name}</div>', unsafe_allow_html=True)
cm = confusion_matrix(y_true, y_pred)

col_cm, col_cr = st.columns([1, 1])
with col_cm:
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    fig_cm.patch.set_facecolor("#f8f9ff")
    ax.set_facecolor("#f8f9ff")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"],
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=10)
    st.pyplot(fig_cm)

with col_cr:
    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, target_names=["Not Fraud", "Fraud"])
    st.code(report, language=None)
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    | | Count |
    |---|---|
    | True Positives (fraud caught) | **{tp}** |
    | False Negatives (fraud missed) | **{fn}** |
    | False Positives (false alarms) | **{fp}** |
    | True Negatives (correct legit) | **{tn}** |
    """)

# ── ROC Curve ─────────────────────────────────────────────────────────────────
st.markdown(f'<div class="section-header">5. ROC Curve — {model_name}</div>', unsafe_allow_html=True)
col_roc, col_dist = st.columns([1, 1])

with col_roc:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig_roc, ax2 = plt.subplots(figsize=(5, 4))
    fig_roc.patch.set_facecolor("#f8f9ff")
    ax2.set_facecolor("#f8f9ff")
    ax2.fill_between(fpr, tpr, alpha=0.15, color="#0f3460")
    ax2.plot(fpr, tpr, color="#0f3460", lw=2.5, label=f"AUC = {metrics['AUC']}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random baseline")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title(f"ROC Curve — {model_name}", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig_roc)

# ── Probability Distribution ──────────────────────────────────────────────────
with col_dist:
    fig_dist, ax3 = plt.subplots(figsize=(5, 4))
    fig_dist.patch.set_facecolor("#f8f9ff")
    ax3.set_facecolor("#f8f9ff")
    ax3.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Not Fraud", color="#2196F3", edgecolor="white")
    ax3.hist(y_prob[y_true == 1], bins=50, alpha=0.8, label="Fraud",     color="#e94560", edgecolor="white")
    ax3.axvline(THRESH, color="#333", linestyle="--", lw=2, label=f"Threshold = {THRESH}")
    ax3.set_xlabel("Predicted Fraud Probability", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("Fraud Probability Distribution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig_dist)

# ── Feature Importance (tree models only) ─────────────────────────────────────
if model_name in ("Random Forest", "Decision Tree"):
    st.markdown(f'<div class="section-header">6. Top 15 Feature Importances — {model_name}</div>', unsafe_allow_html=True)
    clf      = pipeline.named_steps["clf"]
    pre      = pipeline.named_steps["pre"]
    feat_names = numeric_cols + CAT_COLS
    importances = clf.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=True).tail(15)

    fig_fi, ax4 = plt.subplots(figsize=(7, 5))
    fig_fi.patch.set_facecolor("#f8f9ff")
    ax4.set_facecolor("#f8f9ff")
    colors = ["#e94560" if v > fi.median() else "#0f3460" for v in fi.values]
    fi.plot(kind="barh", ax=ax4, color=colors, edgecolor="white")
    ax4.set_xlabel("Importance Score", fontsize=11)
    ax4.set_title(f"Top 15 Feature Importances — {model_name}", fontsize=12, fontweight="bold")
    ax4.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig_fi)
    st.caption("Red bars = above-median importance. Shows which features the model relies on most to detect fraud.")
