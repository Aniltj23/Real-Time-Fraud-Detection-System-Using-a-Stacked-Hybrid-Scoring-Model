import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import shap
import requests
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from scipy.stats import ks_2samp
import lightgbm as lgb
import seaborn as sns
import io

# -------------------------------------------------
# Load model (for SHAP, feature importance)
# -------------------------------------------------
with open("fraud_detection.pkl", "rb") as f:
    model = pickle.load(f)

expected_features = getattr(model, "feature_name_", None)

# -------------------------------------------------
# Load base dataset
# -------------------------------------------------
@st.cache_data
def load_default_data():
    df = pd.read_csv("creditcard.csv")
    return df

base_df = load_default_data()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("⚡ Real-Time Fraud Detection Dashboard (via FastAPI)")
st.write("Streaming live transactions from FastAPI backend with SHAP explanations, drift detection, and leaderboard updates.")

# Sidebar
st.sidebar.header("⚙️ Configuration")
api_url = st.sidebar.text_input("FastAPI Endpoint URL", "http://127.0.0.1:8000/transaction")
threshold = st.sidebar.slider("Fraud Probability Threshold", 0.1, 0.9, 0.5)
n = st.sidebar.slider("Number of transactions to stream", 10, 500, 50)
delay = st.sidebar.slider("Delay between API calls (seconds)", 0.1, 2.0, 0.5)
start_button = st.sidebar.button("🚀 Start Live Stream")

# -------------------------------------------------
# Initialize States
# -------------------------------------------------
if "base_snapshot" not in st.session_state:
    st.session_state.base_snapshot = base_df.sample(5000, random_state=0)
if "drift_history" not in st.session_state:
    st.session_state.drift_history = []
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = pd.DataFrame(columns=["Transaction #", "Fraud Probability", "Actual", "Predicted"])
if "fraud_rate_data" not in st.session_state:
    st.session_state.fraud_rate_data = []

# -------------------------------------------------
# Streaming Section
# -------------------------------------------------
if start_button:
    # Reset session state for a new live run
    st.session_state.fraud_rate_data = []
    st.session_state.leaderboard = pd.DataFrame(columns=["Transaction #", "Fraud Probability", "Actual", "Predicted"])
    st.session_state.drift_history = []
    st.subheader("📡 Streaming Live Transactions from FastAPI...")

    placeholder = st.empty()
    leaderboard_placeholder = st.empty()
    drift_placeholder = st.empty()
    fraud_rate_placeholder = st.empty()

    y_true_all, y_pred_all, probs = [], [], []
    transaction_log = []
    recent_data = []
    fraud_count = 0

    for i in range(n):
        try:
            response = requests.get(api_url, timeout=10)
            if response.status_code != 200:
                st.error(f"❌ API Error: {response.status_code}")
                break

            data = response.json()
            X_display = pd.DataFrame([data["transaction"]])
            y_true = int(data["actual"])
            proba = float(data["fraud_probability"])
            pred = int(bool(data["is_fraud"]))

            if isinstance(y_true, str):
                y_true = 1 if y_true.lower() in ["fraud", "true", "1"] else 0
            if isinstance(pred, str):
                pred = 1 if pred.lower() in ["fraud", "true", "1"] else 0

            available_features = [f for f in expected_features if f in X_display.columns]
            if available_features:
                X_display = X_display[available_features]

        except Exception as e:
            st.error(f"⚠️ Failed to fetch data from API: {e}")
            break

        y_true_all.append(y_true)
        y_pred_all.append(pred)
        probs.append(proba)
        recent_data.append(X_display)

        transaction_log.append({
            "Transaction #": i + 1,
            "Predicted": "Fraud" if pred == 1 else "Legit",
            "Fraud Probability": proba,
            "Actual": "Fraud" if y_true == 1 else "Legit"
        })

        if pred == 1:
            fraud_count += 1

        with placeholder.container():
            st.markdown(f"### Transaction #{i + 1}")
            st.metric("Fraud Probability", f"{proba:.2%}")
            st.write(f"**Predicted:** {'🚨 FRAUD DETECTED' if pred == 1 else '✅ Legitimate'}")
            st.write(f"**Actual:** {'🟥 FRAUD' if y_true == 1 else '🟩 Legitimate'}")
            st.markdown("#### 🧾 Current Transaction Details")
            st.dataframe(X_display.T.reset_index().rename(columns={"index": "Feature", 0: "Value"}), height=180)
            st.markdown(f"**📊 Live Transaction Progress (Transaction #{i + 1})**")
            st.progress((i + 1) / n)
            if pred == 1:
                st.error("⚠️ FRAUD ALERT! Transaction flagged as suspicious.")
        time.sleep(delay)

        # -------------------------------------------------
        # 📈 Fraud Rate Chart (Predicted vs Actual)
        # -------------------------------------------------
        predicted_fraud_rate = (sum(y_pred_all) / len(y_pred_all)) * 100
        actual_fraud_rate = (sum(y_true_all) / len(y_true_all)) * 100

        st.session_state.fraud_rate_data.append({
            "Transaction": i + 1,
            "Predicted Fraud Rate (%)": predicted_fraud_rate,
            "Actual Fraud Rate (%)": actual_fraud_rate
        })

        fraud_rate_df = pd.DataFrame(st.session_state.fraud_rate_data)
        fraud_rate_placeholder.line_chart(fraud_rate_df.set_index("Transaction"), height=200)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            recent_df = pd.concat(recent_data[-500:], ignore_index=True)
            drift_scores = {}
            for col in expected_features[:8]:
                if col in st.session_state.base_snapshot.columns:
                    base_sample = st.session_state.base_snapshot[col].sample(500, random_state=0)
                    new_sample = recent_df[col].sample(min(500, len(recent_df)), random_state=1)
                    stat, p_value = ks_2samp(base_sample, new_sample)
                    drift_scores[col] = p_value
            drift_alerts = [c for c, p in drift_scores.items() if p < 0.05]
            st.session_state.drift_history.append({"Batch": i + 1, "Drifted Features": drift_alerts, "Scores": drift_scores})

        leaderboard_entry = {
            "Transaction #": i + 1,
            "Fraud Probability": proba,
            "Actual": "Fraud" if y_true == 1 else "Legit",
            "Predicted": "Fraud" if pred == 1 else "Legit"
        }
        st.session_state.leaderboard = pd.concat(
            [st.session_state.leaderboard, pd.DataFrame([leaderboard_entry])],
            ignore_index=True
        )
        top10 = st.session_state.leaderboard.sort_values(by="Fraud Probability", ascending=False).head(10)
        leaderboard_placeholder.dataframe(top10, use_container_width=True)

    st.success(f"✅ Streaming complete. {fraud_count} frauds detected out of {n} transactions.")

    st.subheader("🌐 Data Drift Detection Summary")
    if st.session_state.drift_history:
        latest = st.session_state.drift_history[-1]
        drift_df = pd.DataFrame(list(latest["Scores"].items()), columns=["Feature", "p-value"])
        drift_df["Drift Detected"] = drift_df["p-value"] < 0.05
        st.write("**Drift evaluated on the most recent data window**")
        st.dataframe(drift_df)
        if latest["Drifted Features"]:
            st.error(f"⚠️ Drift detected in: {latest['Drifted Features']}")
        else:
            st.success("✅ No significant drift detected.")
    else:
        st.info("No drift data available yet.")

    st.subheader("📊 Model Performance Metrics")
    precision = precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

    def format_metric(value):
        if value == 0 and sum(y_true_all) == 0:
            return "–"
        return f"{value:.3f}"

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", format_metric(precision))
    col2.metric("Recall", format_metric(recall))
    col3.metric("F1 Score", format_metric(f1))

    # Compact Confusion Matrix
    st.markdown("### 🔍 Confusion Matrix")
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(2.3, 2.3))  # small, balanced size
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=ax_cm,
        annot_kws={"size": 7, "weight": "bold"},
        xticklabels=['Pred: Legit', 'Pred: Fraud'],
        yticklabels=['Actual: Legit', 'Actual: Fraud']
    )

    ax_cm.set_xlabel('Predicted', fontsize=5)
    ax_cm.set_ylabel('Actual', fontsize=5)
    ax_cm.tick_params(axis='both', labelsize=4)
    ax_cm.set_title("Confusion Matrix", fontsize=8, pad=4)

    # Center the figure using columns
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.pyplot(fig_cm, use_container_width=False)

    st.subheader("📈 ROC & Precision–Recall Curves")
    fpr, tpr, _ = roc_curve(y_true_all, probs)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true_all, probs)
    pr_auc = auc(rec, prec)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set_title("ROC Curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend()
    ax[1].plot(rec, prec, label=f"AUC = {pr_auc:.3f}", color="orange")
    ax[1].set_title("Precision–Recall Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].legend()
    st.pyplot(fig)

    st.subheader("🏗️ Feature Importance & SHAP Explanation")
    st.markdown("**Global Feature Importance (Model-level)**")
    try:
        fig_imp, ax_imp = plt.subplots(figsize=(6, 6))
        lgb.plot_importance(model, max_num_features=10, ax=ax_imp, importance_type='gain')
        st.pyplot(fig_imp)
    except Exception as e:
        st.warning(f"⚠️ Global feature importance failed: {e}")

    st.markdown("**Local SHAP Explanation (Last Transaction)**")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_display)
        shap_df = pd.DataFrame({
            "Feature": X_display.columns,
            "SHAP Value": shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        }).sort_values(by="SHAP Value", key=abs, ascending=False).head(10)
        st.bar_chart(shap_df.set_index("Feature"))
    except Exception as e:
        st.warning(f"⚠️ SHAP explanation failed: {e}")

    st.subheader("📥 Download Transaction Report")
    report_df = pd.DataFrame(transaction_log)
    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Report", csv, "fraud_report.csv", "text/csv")