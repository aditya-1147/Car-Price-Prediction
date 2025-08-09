import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model (make sure you saved it earlier)
with open("lasso_model.pkl", "rb") as f:
    lass_reg = pickle.load(f)

# Column names from training data
columns = ['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']

st.title("🚗 Car Price Prediction Dashboard")
st.write("Enter the car details below and get the predicted selling price.")

year = st.number_input("Year", min_value=2000, max_value=2025, step=1)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.number_input("Kms Driven", min_value=0, step=1000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.number_input("Owner Count", min_value=0, max_value=3, step=1)

# Encoding
fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_map = {'Dealer': 0, 'Individual': 1}
transmission_map = {'Manual': 0, 'Automatic': 1}

if st.button("Predict Price"):
    input_data = pd.DataFrame([[year, present_price, kms_driven,
                                fuel_map[fuel_type], seller_map[seller_type],
                                transmission_map[transmission], owner]],
                                columns=columns)
    predicted_price = lass_reg.predict(input_data)[0]
    st.success(f"Predicted Selling Price: ₹ {predicted_price:.2f} lakhs")
