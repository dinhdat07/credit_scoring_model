import json
import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# load model
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_credit_default_v3_grid_auc7815.pkl")

model = load_model()

with open("data/mean_values.json", "r") as f:
    mean_values = json.load(f)


feature_info = {
    'LIMIT_BAL': "Amount of given credit in NT dollars",
    'SEX': "1=male, 2=female",
    'EDUCATION': "1=graduate, 2=university, 3=high school, 4=others",
    'MARRIAGE': "1=married, 2=single, 3=others",
    'AGE': "Age in years",
    'PAY_0': "Repayment status in Sept. 2005 (-1=pay duly, 1=1mo late, ..., 9=9+ mos late)",
    'PAY_2': "Repayment status in Aug. 2005",
    'PAY_3': "Repayment status in July. 2005",
    'PAY_4': "Repayment status in June. 2005",
    'PAY_5': "Repayment status in May. 2005",
    'PAY_6': "Repayment status in Apr. 2005",
    'BILL_AMT1': "Bill statement in Sept. 2005",
    'BILL_AMT2': "Bill statement in Aug. 2005",
    'BILL_AMT3': "Bill statement in July. 2005",
    'BILL_AMT4': "Bill statement in June. 2005",
    'BILL_AMT5': "Bill statement in May. 2005",
    'BILL_AMT6': "Bill statement in Apr. 2005",
    'PAY_AMT1': "Payment made in Sept. 2005",
    'PAY_AMT2': "Payment made in Aug. 2005",
    'PAY_AMT3': "Payment made in July. 2005",
    'PAY_AMT4': "Payment made in June. 2005",
    'PAY_AMT5': "Payment made in May. 2005",
    'PAY_AMT6': "Payment made in Apr. 2005"
}

st.title("📊 Credit Default Risk Prediction")
st.markdown("Predict whether a credit card client is likely to default next month.")

st.sidebar.header("Client Information")
def get_user_input():
    input_data = {}
    original_features = list(feature_info.keys())  # exclude engineered features
    for col in original_features:
        info = feature_info.get(col, "")
        val = st.sidebar.number_input(f"{col} ({info})", value=float(mean_values[col]), key=col)
        input_data[col] = val
    return pd.DataFrame([input_data])

input_df = get_user_input()

# feature engineering
def add_features(df):
    df["TOTAL_PAY_AMT"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["TOTAL_BILL_AMT"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["NUM_LATE_PAYMENTS"] = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].apply(lambda row: (row > 0).sum(), axis=1)
    df["MAX_DELAY"] = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].max(axis=1)
    df["LONGEST_LATE_STREAK"] = df[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].apply(longest_streak, axis=1)
    return df

def longest_streak(row):
    streak = 0
    max_streak = 0
    for val in row:
        if val > 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak

input_df = add_features(input_df)

# predict
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    risk_level = "High Risk" if prob > 0.5 else "Low Risk"
    risk_color = "red" if prob > 0.5 else "green"

    st.markdown(f"<h4>Default Probability: <span style='color:{risk_color}'>{prob:.2%} ({risk_level})</span></h4>", unsafe_allow_html=True)
    st.write("Prediction:", "🔴 Default" if pred == 1 else "🟢 No Default")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    st.subheader("Feature Impact (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 3))
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_df.iloc[0], max_display=10, show=False)
    st.pyplot(fig)
