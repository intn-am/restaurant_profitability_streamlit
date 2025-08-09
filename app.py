import streamlit as st
import joblib
import pandas as pd

# -----------------------
# Load Model & Encoders
# -----------------------
model = joblib.load("best_random_forest_model.pkl")
menu_encoder = joblib.load("menu_category_encoder.pkl")
price_scaler = joblib.load("price_scaler.pkl")
profitability_encoder = joblib.load("profitability_encoder.pkl")

# -----------------------
# App Config
# -----------------------
st.set_page_config(
    page_title="Restaurant Profitability Predictor",
    page_icon="🍽️",
    layout="centered"
)

# -----------------------
# Main Title
# -----------------------
st.title("🍽️ Restaurant Menu Profitability Predictor")
st.markdown("""
Aplikasi ini memprediksi **tingkat profitabilitas** menu restoran berdasarkan **kategori menu** dan **harga**.
Silakan masukkan data pada panel di sebelah kiri, lalu klik **Predict**.
""")

# -----------------------
# Sidebar Input
# -----------------------
st.sidebar.header("🔧 Input Data")

# Menu Category Selection (langsung dari encoder)
menu_categories = list(menu_encoder.classes_)
menu_category = st.sidebar.selectbox(
    "Menu Category",
    menu_categories
)

# Price Input
price = st.sidebar.number_input(
    "Price (USD)",
    min_value=0.0,
    step=0.01,
    help="Masukkan harga menu dalam USD"
)

# -----------------------
# Prediction Button
# -----------------------
if st.sidebar.button("Predict"):
    # Encode menu category
    menu_encoded = menu_encoder.transform([menu_category])[0]
    
    # Scale price
    price_scaled = price_scaler.transform([[price]])[0][0]
    
    # Buat DataFrame untuk prediksi
    input_data = pd.DataFrame([[menu_encoded, price_scaled]],
                              columns=['MenuCategory', 'Price'])
    
    # Prediksi
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = profitability_encoder.inverse_transform([prediction_encoded])[0]
    
    # Display Result
    st.subheader("📊 Prediction Result")
    if prediction_label == "High":
        st.success(f"Predicted Profitability: **{prediction_label}** ✅")
        st.markdown("💡 Menu ini berpotensi memberikan **keuntungan tinggi** bagi restoran.")
    elif prediction_label == "Medium":
        st.info(f"Predicted Profitability: **{prediction_label}** ℹ️")
        st.markdown("Menu ini memberikan **keuntungan sedang**. Pertimbangkan strategi harga atau promosi.")
    else:
        st.error(f"Predicted Profitability: **{prediction_label}** ⚠️")
        st.markdown("Menu ini memiliki **keuntungan rendah**. Perlu evaluasi bahan dan harga jual.")

# Footer
st.markdown("---")
st.caption("© 2025 Restaurant Profitability Predictor | Powered by Random Forest Classifier")