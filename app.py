import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_rf.pkl")

st.title("Restaurant Menu Profitability Predictor")

# Input dari user
menu_category = st.selectbox("Menu Category", ["Beverages", "Appetizers", "Desserts", "Main Course"])
price = st.number_input("Price (USD)", min_value=0.0, step=0.01)

# Encode kategori sesuai model training
menu_category_mapping = {
    "Beverages": 0,
    "Appetizers": 1,
    "Desserts": 2,
    "Main Course": 3
}
menu_category_encoded = menu_category_mapping[menu_category]

# Prediksi
if st.button("Predict Profitability"):
    prediction = model.predict([[menu_category_encoded, price]])
    labels = {0: "High", 1: "Low", 2: "Medium"}
    st.success(f"Predicted Profitability: {labels[prediction[0]]}")