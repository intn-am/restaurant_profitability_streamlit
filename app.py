import streamlit as st
import joblib

# =======================
# Load Model & Preprocessing Tools
# =======================
model = joblib.load("best_random_forest_model.pkl")
price_scaler = joblib.load("price_scaler.pkl")
menu_encoder = joblib.load("menu_category_encoder.pkl")
profitability_encoder = joblib.load("profitability_encoder.pkl")

# =======================
# App Configuration
# =======================
st.set_page_config(
    page_title="Restaurant Profitability Predictor",
    page_icon="🍽️",
    layout="wide"
)

# =======================
# Header
# =======================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
    }
    .sub-text {
        text-align: center;
        color: #666;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🍽️ Restaurant Menu Profitability Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Prediksi tingkat profitabilitas menu restoran berdasarkan kategori dan harga.</p>', unsafe_allow_html=True)
st.markdown("---")

# =======================
# Input Form
# =======================
with st.sidebar:
    st.header("🔧 Input Data Menu")

    # Menu Category Selection
    menu_categories = list(menu_encoder.classes_)
    menu_category = st.selectbox(
        "📋 Pilih Kategori Menu:",
        menu_categories
    )

    # Price Input
    price = st.number_input(
        "💲 Harga Menu (USD):",
        min_value=0.0,
        step=0.01,
        help="Masukkan harga menu dalam USD"
    )

    # Predict Button
    predict_button = st.button("🚀 Prediksi Profitabilitas", use_container_width=True)

# =======================
# Prediction
# =======================
if predict_button:
    # Transform input
    menu_encoded = menu_encoder.transform([menu_category])[0]
    price_scaled = price_scaler.transform([[price]])[0][0]

    prediction = model.predict([[menu_encoded, price_scaled]])[0]
    result = profitability_encoder.inverse_transform([prediction])[0]

    # =======================
    # Display Result
    # =======================
    st.subheader("📊 Hasil Prediksi")

    if result == "High":
        st.success(f"✅ Predicted Profitability: **{result}**")
        st.markdown("💡 Menu ini berpotensi memberikan **keuntungan tinggi** bagi restoran. Pertahankan kualitas dan promosi!")
    elif result == "Medium":
        st.info(f"ℹ️ Predicted Profitability: **{result}**")
        st.markdown("📈 Menu ini memberikan **keuntungan sedang**. Pertimbangkan strategi harga atau paket promo untuk meningkatkan penjualan.")
    else:
        st.error(f"⚠️ Predicted Profitability: **{result}**")
        st.markdown("🔍 Menu ini memiliki **keuntungan rendah**. Perlu evaluasi bahan baku, harga jual, atau strategi pemasaran.")

    st.markdown("---")
    st.markdown("### 📌 Detail Input")
    st.write(f"**Kategori Menu:** {menu_category}")
    st.write(f"**Harga (USD):** ${price:.2f}")

# =======================
# Footer
# =======================
st.markdown("---")
st.caption("© 2025 Restaurant Profitability Predictor | Powered by Random Forest Classifier & Streamlit")
