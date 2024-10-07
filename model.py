import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load and preprocess the dataset
data = pd.read_csv("D://projects//orison_tech//Regression//insurance_data.csv")

# Round the 'Premium' column to the nearest integer
data['Premium'] = data['Premium'].round(0).astype(int)

# Select 'Age' as the feature and 'Premium' as the target
X = data[['Age']]  # Select 'Age' column for prediction
y = data["Premium"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Min-Max Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create a Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Streamlit app title
st.title("Insurance Premium Predictor 💼")

# User input for age
user_age = st.number_input("Enter Age (1-120):", min_value=1, max_value=120)

# Predict premium when button is clicked
if st.button('Predict Premium'):
    # Prepare user input data for scaling
    input_data = pd.DataFrame({'Age': [user_age]})
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the predicted premium
    st.success(f"The predicted premium for age {user_age} is: **${prediction[0]:.2f}**")
