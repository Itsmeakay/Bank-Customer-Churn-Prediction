import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Churn Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "bank_churn_model.pkl"
)

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Predict the probability of a customer leaving the bank "
    "and classify the customer into a churn-risk category."
)

st.divider()

# --------------------------------------------------
# SIDEBAR - CUSTOMER INPUT
# --------------------------------------------------

st.sidebar.header("Customer Information")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=40
)

tenure = st.sidebar.number_input(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    max_value=300000.0,
    value=100000.0
)

num_products = st.sidebar.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=1
)

has_card = st.sidebar.selectbox(
    "Has Credit Card",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active = st.sidebar.selectbox(
    "Is Active Member",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=1.0,
    max_value=300000.0,
    value=100000.0
)

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

balance_salary_ratio = (
    balance / estimated_salary
)

log_balance_salary_ratio = np.log1p(
    balance_salary_ratio
)

product_density = (
    num_products / (tenure + 1)
)

age_tenure_interaction = (
    age * (tenure + 1)
)

active_product_interaction = (
    is_active * num_products
)

# --------------------------------------------------
# CUSTOMER DATAFRAME
# --------------------------------------------------

customer = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_products],
    "HasCrCard": [has_card],
    "IsActiveMember": [is_active],
    "EstimatedSalary": [estimated_salary],
    "LogBalanceSalaryRatio": [
        log_balance_salary_ratio
    ],
    "ProductDensity": [
        product_density
    ],
    "AgeTenureInteraction": [
        age_tenure_interaction
    ],
    "ActiveProductInteraction": [
        active_product_interaction
    ]
})

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.sidebar.button("Predict Churn Risk"):

    probability = model.predict_proba(
        customer
    )[0][1]

    # Final threshold selected during model evaluation
    threshold = 0.30

    prediction = (
        probability >= threshold
    )

    # --------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------

    if probability < 0.30:
        risk_category = "Low Risk"
    elif probability < 0.60:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Classification",
            "Likely Churn" if prediction
            else "Likely Retain"
        )

    with col3:
        st.metric(
            "Risk Category",
            risk_category
        )

    st.divider()

    # --------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------

    st.subheader("Churn Risk")

    st.progress(
        float(probability)
    )

    # --------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------

    if risk_category == "Low Risk":

        st.success(
            "🟢 Low Risk — This customer currently "
            "shows relatively low churn probability."
        )

    elif risk_category == "Medium Risk":

        st.warning(
            "🟡 Medium Risk — Consider monitoring "
            "this customer and offering targeted engagement."
        )

    else:

        st.error(
            "🔴 High Risk — This customer has a relatively "
            "high probability of churn. Consider prioritizing "
            "retention actions."
        )

    # --------------------------------------------------
    # CUSTOMER SUMMARY
    # --------------------------------------------------

    st.divider()

    st.subheader("Customer Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(
            f"**Credit Score:** {credit_score}"
        )

        st.write(
            f"**Age:** {age}"
        )

        st.write(
            f"**Geography:** {geography}"
        )

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Tenure:** {tenure} years"
        )

    with summary_col2:

        st.write(
            f"**Balance:** {balance:,.2f}"
        )

        st.write(
            f"**Products:** {num_products}"
        )

        st.write(
            f"**Active Member:** "
            f"{'Yes' if is_active else 'No'}"
        )

        st.write(
            f"**Credit Card:** "
            f"{'Yes' if has_card else 'No'}"
        )

        st.write(
            f"**Estimated Salary:** "
            f"{estimated_salary:,.2f}"
        )

else:

    st.info(
        "Enter customer information in the sidebar "
        "and click **Predict Churn Risk**."
    )

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.caption(
    "Model: Gradient Boosting | "
    "ROC-AUC: 0.8694 | "
    "Classification Threshold: 0.30"
)