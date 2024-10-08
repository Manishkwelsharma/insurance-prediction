# app.py
import streamlit as st
import pandas as pd
from model import load_data, train_model

st.title("Insurance Premium Prediction")

# Load and train model
data = load_data()
model = train_model(data)

# User input for age
age = st.number_input("Enter age", min_value=18, max_value=100, value=25)

# Predict premium
if st.button("Predict Premium"):
    prediction = model.predict([[age]])
    st.write(f"The predicted insurance premium is ${prediction[0]:.2f}")
