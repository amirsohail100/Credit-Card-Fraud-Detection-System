import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

# --- SAFE IMPORTS ---
DEPENDENCIES_LOADED = True
IMPORT_ERR_MSG = ""

try:
    import numpy as np
    import pandas as pd
    import joblib
except ImportError as e:
    DEPENDENCIES_LOADED = False
    IMPORT_ERR_MSG = str(e)

# --- UI STYLING ---
st.markdown("""
    <style>
    .main-title { font-family: 'Inter', sans-serif; color: #1E293B; font-weight: 800; text-align: center; }
    .subtitle { color: #64748B; text-align: center; font-size: 15px; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict fraudulent transactions using PCA features & transaction metrics</p>", unsafe_allow_html=True)

if not DEPENDENCIES_LOADED:
    st.error(f"⚠️ **Missing Dependencies:** {IMPORT_ERR_MSG}")
    st.info("💡 Run: `pip install streamlit scikit-learn numpy pandas joblib`")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    assets = {'model': None, 'scaler': None, 'columns': None, 'errors': []}
    if not DEPENDENCIES_LOADED:
        return assets
    
    for key, filename in [('model', 'model.pkl'), ('scaler', 'scaler.pkl'), ('columns', 'column.pkl')]:
        try:
            assets[key] = joblib.load(filename)
        except Exception as e:
            assets['errors'].append(f"Failed to load '{filename}': {str(e)}")
    return assets

assets = load_assets()

# Status Banner
if DEPENDENCIES_LOADED:
    if assets['model'] is not None and assets['scaler'] is not None:
        st.success("🚀 **System Ready:** Model and Scaler loaded successfully.")
    else:
        st.warning("⚠️ **System Warning:** Model files missing/not found in root folder.")
        for err in assets['errors']:
            st.caption(f"• {err}")

st.markdown("---")
st.markdown("### 📊 Enter Transaction Details")

with st.form("fraud_detection_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        time_val = st.number_input("Time (Seconds)", min_value=0.0, value=121958.0, step=100.0)
        v1 = st.number_input("V1", value=-2.289061, format="%.6f")
        v2 = st.number_input("V2", value=-1.313758, format="%.6f")
        v3 = st.number_input("V3", value=-0.452562, format="%.6f")

    with col2:
        v4 = st.number_input("V4", value=-0.392802, format="%.6f")
        v5 = st.number_input("V5", value=0.224787, format="%.6f")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=1600.89, step=10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("🔍 Detect Fraud", type="primary", use_container_width=True)

# --- PREDICTION LOGIC ---
if submit_btn:
    if not DEPENDENCIES_LOADED:
        st.error("❌ Cannot execute: Required python libraries missing.")
    elif assets['model'] is None:
        st.error("❌ Cannot execute: `model.pkl` is missing.")
    else:
        try:
            raw_inputs = np.array([[time_val, v1, v2, v3, v4, v5, amount]])
            
            # Apply scaling if scaler is available
            if assets['scaler'] is not None:
                features_to_predict = assets['scaler'].transform(raw_inputs)
            else:
                features_to_predict = raw_inputs

            prediction = assets['model'].predict(features_to_predict)[0]
            
            st.markdown("---")
            if int(prediction) == 1:
                st.error("🚨 **ALERT: High Risk!** This transaction is flagged as **FRAUDULENT (Class 1)**.")
            else:
                st.success("✅ **SAFE:** This transaction appears to be **LEGITIMATE (Class 0)**.")
                
        except Exception as pred_err:
            st.error(f"❌ Inference Error: {str(pred_err)}")