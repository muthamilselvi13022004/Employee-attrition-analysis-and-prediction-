
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification



## Streamlit UI

st.set_page_config(page_title="Employee Attrition Analysis and Prediction",layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Mini Project 3"])
#------------------------------------------------------------------------------------

X, y = make_classification(n_samples=200, n_features=5, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)


#  Streamlit

st.title("Employee Attrition Analysis and Prediction")
st.markdown("HR Analytics: Predicting and Preventing Employee Attrition")
st.write("Enter employee details below to predict Attrition (Yes/No):")

## load data
data = pd.read_csv(r"C:\Users\manju\Downloads\Employee-Attrition - Employee-Attrition.csv")
st.write("Here is the dataset")
st.dataframe(data)

#dataset features
age = st.number_input("Age", min_value=18, max_value=60, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.selectbox("Overtime", ["Yes", "No"])

# Convert categorical input
overtime_val = 1 if overtime == "Yes" else 0

# Collect into DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "JobSatisfaction": [job_satisfaction],
    "YearsAtCompany": [years_at_company],
    "Overtime": [overtime_val]
})

# Initialize session state to store all predictions
if "employee_data" not in st.session_state:
    st.session_state["employee_data"] = pd.DataFrame(columns=["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "Overtime", "Prediction"])

## Prediction
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    pred_label = "Yes" if prediction == 1 else "No"

    # Add prediction to the input data
    input_data["Prediction"] = pred_label

    # Store in session state
    st.session_state["employee_data"] = pd.concat([st.session_state["employee_data"], input_data], ignore_index=True)

    # Show message
    if prediction == 1:
        st.error("This employee is leave the company  Attrition is (Yes).")
    else:
        st.success(" This employee is not leave the company  Attrition is (No).")


st.write("### All Employee Data with Predictions")
st.dataframe(st.session_state["employee_data"])
