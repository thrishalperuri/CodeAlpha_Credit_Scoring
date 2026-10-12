import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Scoring & Risk Assessment | CodeAlpha",
    page_icon="💳",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.pkl")

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
    }
    .approved-card {
        background-color: #ECFDF5;
        border: 1.5px solid #10B981;
        padding: 1.2rem;
        border-radius: 10px;
        color: #065F46;
    }
    .denied-card {
        background-color: #FEF2F2;
        border: 1.5px solid #EF4444;
        padding: 1.2rem;
        border-radius: 10px;
        color: #991B1B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">💳 Credit Scoring & Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CodeAlpha Machine Learning Internship Project — Predict individual creditworthiness using financial history.</div>', unsafe_allow_html=True)

# Check if model exists
@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

artifact = load_trained_model()

if artifact is None:
    st.warning("⚠️ Trained model not found! Please run `python train_model.py` first to train and serialize the model.")
    if st.button("Train Model Now"):
        with st.spinner("Training models and generating benchmark charts..."):
            import subprocess
            subprocess.run(["python", os.path.join(BASE_DIR, "train_model.py")], check=True)
            st.success("Model trained successfully! Please refresh the page.")
            st.rerun()

st.sidebar.header("📋 Applicant Financial Profile")

age = st.sidebar.slider("Age", min_value=18, max_value=75, value=35, step=1)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=300000, value=65000, step=2500)
employment_years = st.sidebar.slider("Employment History (Years)", min_value=0.0, max_value=40.0, value=6.5, step=0.5)
total_debt = st.sidebar.number_input("Existing Total Debt ($)", min_value=0, max_value=250000, value=18000, step=1000)

dti_calc = round(total_debt / max(annual_income, 1), 3)
st.sidebar.info(f"📊 Calculated Debt-to-Income (DTI) Ratio: **{dti_calc:.1%}**")

open_credit_lines = st.sidebar.slider("Number of Open Credit Accounts", min_value=1, max_value=20, value=6)
delinquent_accounts = st.sidebar.selectbox("Past Delinquencies / Defaults", options=[0, 1, 2, 3, 4, 5], index=0)
credit_score = st.sidebar.slider("Credit Score (FICO)", min_value=300, max_value=850, value=710, step=5)
loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=1000, max_value=60000, value=15000, step=1000)
loan_purpose = st.sidebar.selectbox("Loan Purpose", options=[
    "Debt Consolidation", "Home Improvement", "Education", "Business", "Personal", "Auto"
])

# Layout: Split into 2 columns
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📊 Applicant Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Credit Score", f"{credit_score}", "Good" if credit_score >= 670 else "Poor")
    c2.metric("Annual Income", f"${annual_income:,.0f}")
    c3.metric("DTI Ratio", f"{dti_calc:.1%}", "-Healthy" if dti_calc < 0.40 else "High Risk")

    st.write("---")
    st.subheader("🎯 Evaluation & Model Inference")

    if st.button("Run Credit Assessment", type="primary", use_container_width=True):
        if artifact is not None:
            model = artifact["model"]
            scaler = artifact["scaler"]
            encoder = artifact["encoder"]
            num_cols = artifact["numeric_cols"]
            cat_cols = artifact["categorical_cols"]

            # Construct single-row DataFrame
            input_dict = {
                "Age": [age],
                "Annual_Income": [annual_income],
                "Employment_Years": [employment_years],
                "Total_Debt": [total_debt],
                "DTI_Ratio": [dti_calc],
                "Open_Credit_Lines": [open_credit_lines],
                "Delinquent_Accounts": [delinquent_accounts],
                "Credit_Score": [credit_score],
                "Loan_Amount": [loan_amount],
                "Loan_Purpose": [loan_purpose]
            }
            input_df = pd.DataFrame(input_dict)

            # Preprocess
            num_scaled = scaler.transform(input_df[num_cols])
            cat_encoded = encoder.transform(input_df[cat_cols])
            X_input = np.hstack([num_scaled, cat_encoded])

            # Predict
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0][1] if hasattr(model, "predict_proba") else 0.5

            st.write("### Decision Result:")
            if pred == 1:
                st.markdown(
                    f"""
                    <div class="approved-card">
                        <h3>✅ APPROVED (Creditworthy)</h3>
                        <p>Confidence / Approval Probability: <strong>{prob * 100:.1f}%</strong></p>
                        <p>Model used: <strong>{artifact.get('model_name', 'Ensemble ML')}</strong></p>
                        <p>The applicant meets financial stability thresholds with an acceptable credit risk profile.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="denied-card">
                        <h3>❌ REJECTED / HIGH RISK</h3>
                        <p>Default Risk Probability: <strong>{(1 - prob) * 100:.1f}%</strong></p>
                        <p>Model used: <strong>{artifact.get('model_name', 'Ensemble ML')}</strong></p>
                        <p>The applicant exhibits elevated financial risk factors (e.g., high DTI ratio, lower credit score, or past delinquencies).</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.error("Please run training first to build the model.")

with col_right:
    st.subheader("📈 Model Performance Charts")
    plot_cm = os.path.join(BASE_DIR, "plots", "confusion_matrix.png")
    plot_roc = os.path.join(BASE_DIR, "plots", "roc_curve_comparison.png")
    plot_fi = os.path.join(BASE_DIR, "plots", "feature_importance.png")

    tabs = st.tabs(["ROC Curves", "Confusion Matrix", "Feature Importance"])
    with tabs[0]:
        if os.path.exists(plot_roc):
            st.image(plot_roc, caption="Model Comparison ROC Curves", use_container_width=True)
        else:
            st.info("Train the model to generate the ROC-AUC comparison plot.")
    with tabs[1]:
        if os.path.exists(plot_cm):
            st.image(plot_cm, caption="Confusion Matrix on Test Dataset", use_container_width=True)
        else:
            st.info("Train the model to view the confusion matrix.")
    with tabs[2]:
        if os.path.exists(plot_fi):
            st.image(plot_fi, caption="Key Financial Risk Drivers", use_container_width=True)
        else:
            st.info("Train the model to view top feature importances.")
