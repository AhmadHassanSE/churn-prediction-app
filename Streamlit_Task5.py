import streamlit as st
import pandas as pd
import pickle

# 1. Load the saved pipeline
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction App")

# 2. Take User Input
tenure = st.number_input("Tenure (months)", min_value=0, value=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=95.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=95.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# 3. Prediction logic
if st.button("Predict Churn"):
    # Create dataframe for the single input
    input_data = pd.DataFrame([[tenure, monthly_charges, total_charges, contract, internet_service]], 
                              columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of Churning

    if prediction == 1:
        st.error("Prediction: Customer will Churn")
    else:
        st.success("Prediction: Customer will Stay")
        
    st.write(f"Churn Probability Score: {probability:.2f}")
