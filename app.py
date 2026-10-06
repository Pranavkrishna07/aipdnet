
import os, json, time, re
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px

from model_utils import (
    load_or_train, extract_url_features, get_feature_names,
    predict_url, calculate_risk
)

st.set_page_config(
    page_title="AIPDNet | Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = "models"
BASE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE, "scan_history.csv")

# ---------- Styling ----------
st.markdown("""
<style>
.main {background:#f7f9fc;}
.block-container {padding-top:2rem;}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg,#0f172a,#1e3a8a);
    color:white;
    margin-bottom:22px;
}
.hero h1 {font-size:42px;margin-bottom:5px;}
.hero p {font-size:17px;color:#dbeafe;}
.metric-card {
    padding:18px;border-radius:14px;background:white;
    border:1px solid #e5e7eb;box-shadow:0 2px 8px rgba(0,0,0,.04);
}
.safe {padding:18px;border-radius:14px;background:#ecfdf5;border:1px solid #a7f3d0;}
.warn {padding:18px;border-radius:14px;background:#fffbeb;border:1px solid #fde68a;}
.danger {padding:18px;border-radius:14px;background:#fef2f2;border:1px solid #fecaca;}
.small {font-size:13px;color:#64748b;}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "timestamp","url","prediction","risk_level","risk_score","probability"
    ])

def save_scan(row):
    df = load_history()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def result_box(level, score, probability):
    if level == "SAFE":
        st.markdown(
            f'<div class="safe"><h2>🟢 SAFE</h2>'
            f'<b>Website Risk Index:</b> {score}/100<br>'
            f'<b>Phishing probability:</b> {probability:.2%}<br>'
            f'No strong phishing indicators were detected by the current model.</div>',
            unsafe_allow_html=True)
    elif level == "SUSPICIOUS":
        st.markdown(
            f'<div class="warn"><h2>🟡 SUSPICIOUS</h2>'
            f'<b>Website Risk Index:</b> {score}/100<br>'
            f'<b>Phishing probability:</b> {probability:.2%}<br>'
            f'Use caution. Verify the domain before entering credentials.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="danger"><h2>🔴 HIGH RISK</h2>'
            f'<b>Website Risk Index:</b> {score}/100<br>'
            f'<b>Phishing probability:</b> {probability:.2%}<br>'
            f'Do not enter passwords, OTPs, card details, or other sensitive information.</div>',
            unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.title("🛡️ AIPDNet")
page = st.sidebar.radio(
    "Navigation",
    ["URL Scanner","Dashboard","Model & Explainability","Scan History","About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("AIPDNet • AI-Based Phishing Detection")
if st.sidebar.button("⚙️ Train / Refresh Model"):
    with st.spinner("Downloading/training models..."):
        load_or_train(force=True)
        st.cache_resource.clear()
    st.sidebar.success("Model ready. Restart scan if needed.")

# ---------- Model ----------
@st.cache_resource(show_spinner="Loading models (first run downloads the dataset and trains, about 1-3 minutes)...")
def get_bundle():
    return load_or_train(force=False)

try:
    bundle = get_bundle()
except Exception as e:
    bundle = None
    st.error(f"Model initialization failed: {e}")
    st.info("Click 'Train / Refresh Model' in the sidebar after installing requirements.")

# ---------- Home ----------
if page == "URL Scanner":
    st.markdown("""
    <div class="hero">
        <h1>🛡️ AIPDNet</h1>
        <p>Intelligent AI-Based Phishing Website Detection and Risk Assessment Framework</p>
        <p class="small">Analyze a URL using ensemble machine learning and receive an interpretable security risk score.</p>
    </div>
    """, unsafe_allow_html=True)

    if bundle is None:
        st.stop()

    url = st.text_input(
        "Enter website URL",
        placeholder="https://example.com",
        help="Enter a complete URL including https:// or http://"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Models", "3 + Ensemble")
    with c2:
        st.metric("Risk Levels", "3")
    with c3:
        st.metric("Explainability", "SHAP")

    if st.button("🔍 Scan Website", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Extracting URL features and evaluating risk..."):
                    features, feature_info = extract_url_features(url, bundle["feature_names"])
                    pred, probability = predict_url(bundle, features)
                    score, level = calculate_risk(probability)
                st.session_state["last_scan"] = {
                    "url": url, "features": features,
                    "feature_info": feature_info,
                    "prediction": pred, "probability": probability,
                    "score": score, "level": level
                }
                save_scan({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "url": url,
                    "prediction": "PHISHING" if pred == 1 else "LEGITIMATE",
                    "risk_level": level,
                    "risk_score": score,
                    "probability": round(probability, 6)
                })
            except Exception as e:
                st.error(f"Scan failed: {e}")

    scan = st.session_state.get("last_scan")
    if scan:
        st.markdown("---")
        st.subheader("Detection Result")
        st.write(f"**URL:** `{scan['url']}`")
        result_box(scan["level"], scan["score"], scan["probability"])

        a,b,c = st.columns(3)
        with a:
            st.metric("Risk Score", f"{scan['score']}/100")
        with b:
            st.metric("Phishing Probability", f"{scan['probability']:.2%}")
        with c:
            st.metric("Classification", "PHISHING" if scan["prediction"] else "LEGITIMATE")

        st.subheader("Security Recommendation")
        if scan["level"] == "HIGH RISK":
            st.error("Avoid this website and do not submit credentials, payment details, OTPs, or personal information.")
        elif scan["level"] == "SUSPICIOUS":
            st.warning("Verify the domain through an independent source before interacting with the website.")
        else:
            st.success("The model found no strong phishing signal. Still verify unexpected links before entering sensitive information.")

        st.subheader("Extracted URL Features")
        fi = pd.DataFrame({
            "Feature": list(scan["feature_info"].keys()),
            "Value": [str(v) for v in scan["feature_info"].values()]
        })
        st.dataframe(fi, hide_index=True)

# ---------- Dashboard ----------
elif page == "Dashboard":
    st.title("📊 Security Dashboard")
    hist = load_history()
    if hist.empty:
        st.info("No scans yet. Use URL Scanner to create scan history.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Total Scans", len(hist))
        with c2: st.metric("High Risk", int((hist.risk_level=="HIGH RISK").sum()))
        with c3: st.metric("Suspicious", int((hist.risk_level=="SUSPICIOUS").sum()))
        with c4: st.metric("Safe", int((hist.risk_level=="SAFE").sum()))

        counts = hist["risk_level"].value_counts().reset_index()
        counts.columns = ["Risk Level","Count"]
        fig = px.bar(counts, x="Risk Level", y="Count", title="Risk Distribution")
        st.plotly_chart(fig)

        fig2 = px.histogram(hist, x="risk_score", nbins=20, title="Website Risk Score Distribution")
        st.plotly_chart(fig2)

# ---------- Model ----------
elif page == "Model & Explainability":
    st.title("🤖 Model & Explainability")
    if bundle is None:
        st.stop()

    st.write("AIPDNet combines Random Forest, XGBoost and LightGBM using soft-voting ensemble learning.")

    metrics_path = os.path.join(BASE, "results", "model_comparison.csv")
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        st.subheader("Model Performance")
        st.dataframe(metrics, hide_index=True)

        numeric = [c for c in ["Accuracy","Precision","Recall","F1","ROC-AUC"] if c in metrics.columns]
        if numeric:
            melted = metrics.melt(id_vars=["Model"], value_vars=numeric,
                                  var_name="Metric", value_name="Score")
            fig = px.bar(melted, x="Model", y="Score", color="Metric",
                         barmode="group", range_y=[0,1],
                         title="Model Comparison")
            st.plotly_chart(fig)

    st.subheader("Feature Importance / SHAP")
    shap_path = os.path.join(BASE, "results", "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP feature importance")
    else:
        st.info("Train the model to generate SHAP analysis.")

# ---------- History ----------
elif page == "Scan History":
    st.title("🕘 Scan History")
    hist = load_history()
    if hist.empty:
        st.info("No scan history available.")
    else:
        st.dataframe(hist.sort_values("timestamp", ascending=False), hide_index=True)
        st.download_button(
            "⬇️ Download Scan History CSV",
            hist.to_csv(index=False).encode("utf-8"),
            "aipdnet_scan_history.csv",
            "text/csv"
        )

# ---------- About ----------
else:
    st.title("ℹ️ About AIPDNet")
    st.markdown("""
### AIPDNet
**An Intelligent AI-Based Phishing Website Detection and Risk Assessment Framework**

AIPDNet is a software-based cybersecurity system designed to detect potentially
phishing websites using machine-learning techniques.

### Core pipeline

**URL → Feature Extraction → Random Forest / XGBoost / LightGBM → Soft Voting → Phishing Probability → WRI → Risk Level → Explanation**

### Technologies
- Python
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Plotly

### Risk interpretation

| WRI | Level |
|---:|---|
| 0–30 | SAFE |
| 31–60 | SUSPICIOUS |
| 61–100 | HIGH RISK |

### Disclaimer
This project is an academic research prototype. A model prediction is not a guarantee
that a website is safe or malicious. Do not use it as the sole basis for cybersecurity decisions.
""")
