import streamlit as st
import pandas as pd
import numpy as np
import joblib


def app():
    st.set_page_config(page_title="🔍 Credit Risk Checker", layout="wide")
    st.title("🔐 Credit Risk Prediction App")

    st.markdown("Welcome to the **Credit Risk Checker**! Fill out the details below to evaluate the risk level of a credit application.")

    st.markdown("---")

    # Layout with columns
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("📅 Age", 18, 100, 35)

        sex = st.radio("🧍 Gender", ["Male", "Female"], horizontal=True)

        job = st.selectbox(
            "💼 Employment Status",
            [
                "Unskilled and Non-resident",
                "Unskilled and Resident",
                "Skilled",
                "Highly Skilled",
            ],
        )

    with col2:
        housing = st.radio("🏠 Housing Type", ["Own", "Free", "Rent"], horizontal=True).lower()

        saving_accounts = st.selectbox(
            "💰 Savings Account Balance",
            [
                "Not Available",
                "Little",
                "Moderate",
                "Quite Rich",
                "Rich",
            ],
        )

        checking_account = st.selectbox(
            "🏦 Checking Account Balance",
            [
                "Not Available",
                "Little",
                "Moderate",
                "Rich",
            ],
        )

    with col3:
        credit_amount = st.number_input(
            "💸 Credit Amount Requested",
            min_value=0,
            max_value=20000,
            value=700,
            step=100,
            format="%d",
        )

        duration = st.number_input(
            "⏳ Duration of Loan (Months)",
            min_value=0,
            max_value=100,
            value=12,
            step=1,
            format="%d",
        )

        purpose = st.selectbox(
            "🎯 Purpose of the Loan",
            [
                "car",
                "furniture/equipment",
                "radio/TV",
                "domestic appliances",
                "repairs",
                "education",
                "business",
                "vacation/others",
            ],
        )

    st.markdown("---")

    if st.button("🚀 Predict Credit Risk"):
        prediction = predict(
            {
                "Age": age,
                "Sex": sex,
                "Job": job,
                "Housing": housing,
                "Saving accounts": saving_accounts,
                "Checking account": checking_account,
                "Credit amount": credit_amount,
                "Duration": duration,
                "Purpose": purpose,
            }
        )

        st.success("✅ Prediction completed!")

        if prediction == 0:
            st.error("⚠️ Credit Risk **Detected**")
        else:
            st.success("🎉 No Credit Risk Found")


def predict(d):
    df = pd.DataFrame(
        {
            "Age": np.log(d["Age"]),
            "Job": {
                "Unskilled and Non-resident": 0,
                "Unskilled and Resident": 1,
                "Skilled": 2,
                "Highly Skilled": 3,
            }[d["Job"]],
            "Credit amount": np.log(d["Credit amount"]),
            "Duration": np.log(d["Duration"]),
            "Sex_female": d["Sex"] == "Female",
            "Sex_male": d["Sex"] == "Male",
            "Checking account_Not Available": d["Checking account"]
            == "Not Available",
            "Checking account_little": d["Checking account"] == "Little",
            "Checking account_moderate": d["Checking account"] == "Moderate",
            "Checking account_rich": d["Checking account"] == "Rich",
        },
        index=["row1"],
    )

    model = joblib.load("./artifacts/model.pkl")
    return model.predict(df)


if __name__ == "__main__":
    app()
