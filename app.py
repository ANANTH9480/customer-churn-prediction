import streamlit as st
import pandas as pd
import joblib


model = joblib.load(r"C:\Users\DELL\Customer_churn_prediction\customer_churn_model.pkl")
scaler = joblib.load(r"C:\Users\DELL\Customer_churn_prediction\scaler.pkl")
columns = joblib.load(r"C:\Users\DELL\Customer_churn_prediction\columns.pkl")

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to churn based on service and billing information.")


st.sidebar.header("Model Performance")
st.sidebar.write("Accuracy: 82%")
st.sidebar.write("Algorithm: Logistic Regression")


gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["No", "Yes"])
Dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.number_input("Tenure", min_value=0)

PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")

if st.button("Predict"):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=columns
    )

    # Numeric columns
    input_data["SeniorCitizen"] = SeniorCitizen
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = MonthlyCharges
    input_data["TotalCharges"] = TotalCharges

    # Binary dummies
    if gender == "Male":
        input_data["gender_Male"] = 1

    if Partner == "Yes":
        input_data["Partner_Yes"] = 1

    if Dependents == "Yes":
        input_data["Dependents_Yes"] = 1

    if PhoneService == "Yes":
        input_data["PhoneService_Yes"] = 1

    # Multiple Lines
    if MultipleLines == "Yes":
        input_data["MultipleLines_Yes"] = 1
    elif MultipleLines == "No phone service":
        input_data["MultipleLines_No phone service"] = 1

    # Internet Service
    if InternetService == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif InternetService == "No":
        input_data["InternetService_No"] = 1

    # Online Security
    if OnlineSecurity == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif OnlineSecurity == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1

    # Online Backup
    if OnlineBackup == "Yes":
        input_data["OnlineBackup_Yes"] = 1
    elif OnlineBackup == "No internet service":
        input_data["OnlineBackup_No internet service"] = 1

    # Device Protection
    if DeviceProtection == "Yes":
        input_data["DeviceProtection_Yes"] = 1
    elif DeviceProtection == "No internet service":
        input_data["DeviceProtection_No internet service"] = 1

    # Tech Support
    if TechSupport == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif TechSupport == "No internet service":
        input_data["TechSupport_No internet service"] = 1

    # Streaming TV
    if StreamingTV == "Yes":
        input_data["StreamingTV_Yes"] = 1
    elif StreamingTV == "No internet service":
        input_data["StreamingTV_No internet service"] = 1

    # Streaming Movies
    if StreamingMovies == "Yes":
        input_data["StreamingMovies_Yes"] = 1
    elif StreamingMovies == "No internet service":
        input_data["StreamingMovies_No internet service"] = 1

    # Contract
    if Contract == "One year":
        input_data["Contract_One year"] = 1
    elif Contract == "Two year":
        input_data["Contract_Two year"] = 1

    # Paperless Billing
    if PaperlessBilling == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    # Payment Method
    if PaymentMethod == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif PaymentMethod == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif PaymentMethod == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1

    st.write("Input Data Sent To Model")
    st.write(input_data)

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        st.error("Customer Will Churn")
    else:
        st.success("Customer Will Not Churn")
    
    st.markdown("---")
    st.markdown("Developed by Ananth Reddy")