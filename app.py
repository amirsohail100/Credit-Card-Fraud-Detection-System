import streamlit as st

# --- 1. PAGE CONFIGURATION (Always Renders First) ---
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

# --- 2. SAFE DEPENDENCY CHECK & IMPORTS ---
DEPENDENCIES_LOADED = True
MISSING_IMPORT_MSG = ""

try:
    import numpy as np
    import pandas as pd
    import joblib
except ImportError as err:
    DEPENDENCIES_LOADED = False
    MISSING_IMPORT_MSG = str(err)

# --- 3. CUSTOM UI STYLING ---
st.markdown("""
    <style>
    .main-title { 
        font-family: 'Inter', sans-serif; 
        color: #1E293B; 
        font-weight: 800; 
        text-align: center; 
        margin-bottom: 5px;
    }
    .subtitle { 
        color: #64748B; 
        text-align: center; 
        font-size: 15px; 
        margin-bottom: 25px; 
    }
    .stButton>button {
        border-radius: 8px;
        height: 48px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title and Subtitle Header
st.markdown("<h1 class='main-title'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict fraudulent transactions using PCA features & transaction metrics</p>", unsafe_allow_html=True)

# Display Dependency Error Message if Imports Failed
if not DEPENDENCIES_LOADED:
    st.error(f"⚠️ **Missing Dependencies Detected:** {MISSING_IMPORT_MSG}")
    st.info("💡 Please install missing dependencies using: `pip install streamlit scikit-learn numpy pandas joblib`")

# --- 4. SAFE MODEL & ASSET LOADING FUNCTION ---
@st.cache_resource
def load_all_assets():
    """
    Safely loads all pickle files using joblib with individual try-except blocks.
    Ensures that a single missing file does not break the entire app.
    """
    assets = {
        'model': None,
        'scaler': None,
        'columns': None,
        'errors': []
    }
    
    if not DEPENDENCIES_LOADED:
        return assets

    # Safely load model.pkl
    try:
        assets['model'] = joblib.load('final_model.pkl')
    except Exception as e:
        assets['errors'].append(f"Failed to load 'model.pkl': {str(e)}")

    # Safely load scaler.pkl
    try:
        assets['scaler'] = joblib.load('scaler.pkl')
    except Exception as e:
        assets['errors'].append(f"Failed to load 'scaler.pkl': {str(e)}")

    # Safely load column.pkl / columns.pkl
    try:
        assets['columns'] = joblib.load('columns.pkl')
    except Exception as e:
        try:
            assets['columns'] = joblib.load('columns.pkl')
        except Exception as inner_e:
            assets['errors'].append(f"Failed to load 'column.pkl': {str(inner_e)}")

    return assets

assets = load_all_assets()

# Status Banner Display
st.markdown("---")
if DEPENDENCIES_LOADED:
    if assets['model'] is not None:
        st.success("🚀 **System Ready:** Classification model and resources loaded successfully.")
    else:
        st.warning("⚠️ **Model Loading Warning:** Unable to load all model files from the root directory.")
        for err_msg in assets['errors']:
            st.caption(f"• {err_msg}")

# --- 5. INPUT FORM (Guaranteed to Render Always) ---
st.markdown("### 📊 Enter Transaction Details")

with st.form("credit_card_fraud_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        time_val = st.number_input(
            "Transaction Timestamp (Seconds)", 
            min_value=0.0, 
            value=121958.0, 
            step=100.0,
            help="Elapsed time in seconds since the first recorded transaction in the system."
        )
        # Displaying meaningful label to user, assigning to 'v1' for model
        v1 = st.number_input(
            "Transaction Behavior Factor 1 (V1)", 
            value=-2.289061, 
            format="%.6f",
            help="PCA component representing core transaction behavior analysis."
        )
        v2 = st.number_input(
            "Account Activity Vector 2 (V2)", 
            value=-1.313758, 
            format="%.6f",
            help="PCA component capturing account usage and transaction patterns."
        )
        v3 = st.number_input(
            "Security Risk Index 3 (V3)", 
            value=-0.452562, 
            format="%.6f",
            help="PCA component measuring transaction security variance."
        )

    with col2:
        v4 = st.number_input(
            "Anomalous Pattern Score 4 (V4)", 
            value=-0.392802, 
            format="%.6f",
            help="PCA component identifying deviation from normal spending behavior."
        )
        v5 = st.number_input(
            "Location/Device Metric 5 (V5)", 
            value=0.224787, 
            format="%.6f",
            help="PCA component evaluating contextual transaction parameters."
        )
        amount = st.number_input(
            "Transaction Amount ($)", 
            min_value=0.0, 
            value=1600.89, 
            step=10.0,
            help="Total monetary value of the transaction."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("🔍 Detect Fraudulent Activity", type="primary", use_container_width=True)
# --- 6. PREDICTION & INFERENCE EXECUTION ---
if submit_btn:
    # Check if dependencies are missing
    if not DEPENDENCIES_LOADED:
        st.error("❌ Prediction aborted: Required Python libraries are not installed.")
    # Check if the primary trained model is missing
    elif assets['model'] is None:
        st.error("❌ Prediction aborted: The file `model.pkl` is missing or invalid.")
    else:
        try:
            # 1. Assemble raw input values into array
            raw_input_data = np.array([[time_val, v1, v2, v3, v4, v5, amount]])
            
            # 2. Apply feature scaling if scaler exists
            if assets['scaler'] is not None:
                try:
                    features_for_prediction = assets['scaler'].transform(raw_input_data)
                except Exception as scale_err:
                    # Fallback to unscaled input if feature count mismatches
                    st.warning(f"⚠️ Scaler warning: {str(scale_err)}. Proceeding without scaling.")
                    features_for_prediction = raw_input_data
            else:
                features_for_prediction = raw_input_data

            # 3. Predict transaction class
            prediction_result = assets['model'].predict(features_for_prediction)[0]
            
            # 4. Render prediction output card
            st.markdown("---")
            st.markdown("### 🏆 Prediction Result")
            
            if int(prediction_result) == 1:
                st.error("🚨 **HIGH RISK DETECTED:** This transaction is classified as **FRAUDULENT (Class 1)**.")
            else:
                st.success("✅ **SAFE TRANSACTION:** This transaction is classified as **LEGITIMATE (Class 0)**.")
                
        except Exception as prediction_error:
            st.error(f"❌ An error occurred during prediction inference: {str(prediction_error)}")

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; font-size: 13px;'>Credit Card Fraud Detection System | Fault-Tolerant Web Pipeline</p>", unsafe_allow_html=True)