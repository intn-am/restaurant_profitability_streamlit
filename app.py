import streamlit as st
import joblib
import pandas as pd

# -----------------------
# Load model dan encoder/scaler
# -----------------------
model = joblib.load("best_random_forest_model.pkl")

# Load encoder kategori menu
menu_encoder = joblib.load("menu_category_encoder.pkl")

# Load scaler harga
price_scaler = joblib.load("price_scaler.pkl")

# Load encoder label profitabilitas
profitability_encoder = joblib.load("profitability_encoder.pkl")

# -----------------------
# Pastikan kategori menu terbaca
# -----------------------
if isinstance(menu_encoder, dict):
    menu_categories = list(menu_encoder.keys())
else:
    menu_categories = list(menu_encoder.classes_)

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

# Menu Category Selection
menu_category = st.sidebar.selectbox("Menu Category", menu_categories)

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
    # Encode kategori
    if isinstance(menu_encoder, dict):
        menu_encoded = menu_encoder[menu_category]
    else:
        menu_encoded = menu_encoder.transform([menu_category])[0]
    
    # Scale harga
    price_scaled = price_scaler.transform([[price]])[0][0]
    
    # Prediksi
    prediction = model.predict([[menu_encoded, price_scaled]])[0]
    
    # Decode hasil prediksi
    if isinstance(profitability_encoder, dict):
        result = [k for k, v in profitability_encoder.items() if v == prediction][0]
    else:
        result = profitability_encoder.inverse_transform([prediction])[0]
    
    # Display Result
    st.subheader("📊 Prediction Result")
    if result.lower() == "high":
        st.success(f"Predicted Profitability: **{result}** ✅")
        st.markdown("💡 Menu ini berpotensi memberikan **keuntungan tinggi** bagi restoran.")
    elif result.lower() == "medium":
        st.info(f"Predicted Profitability: **{result}** ℹ️")
        st.markdown("Menu ini memberikan **keuntungan sedang**. Pertimbangkan strategi harga atau promosi.")
    else:
        st.error(f"Predicted Profitability: **{result}** ⚠️")
        st.markdown("Menu ini memiliki **keuntungan rendah**. Perlu evaluasi bahan dan harga jual.")

# Footer
st.markdown("---")
st.caption("© 2025 Restaurant Profitability Predictor | Powered by Random Forest Classifier")