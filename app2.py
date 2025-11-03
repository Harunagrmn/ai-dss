import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import time

# =========================================
# PAGE SETTINGS
# =========================================
st.set_page_config(
    page_title="KârMentor | AI Profit Assistant",
    layout="wide",
    page_icon="💼"
)

# =========================================
# SESSION VARS
# =========================================
if "lang" not in st.session_state:
    st.session_state.lang = "Türkçe"
if "dark" not in st.session_state:
    st.session_state.dark = False
if "model" not in st.session_state:
    st.session_state.model = None
if "veri" not in st.session_state:
    st.session_state.veri = pd.DataFrame([
        [100, 60, 20, 0.10, 0.40],
        [120, 70, 25, 0.05, 0.42],
        [150, 80, 30, 0.00, 0.50],
        [90,  55, 18, 0.15, 0.33],
        [130, 75, 28, 0.07, 0.46],
    ], columns=["Birim Fiyat", "Birim Maliyet", "Satış Adedi", "İndirim Oranı", "Kâr Oranı"])

# =========================================
# SIDEBAR (ICON + ANIMATION)
# =========================================
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #012C2E, #024C4F, #026D71);
    animation: gradientShift 10s ease infinite;
    background-size: 400% 400%;
    color: white;
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
    color: white !important;
}
.sidebar-title {
    text-align:center; 
    font-size:22px; 
    font-weight:bold; 
    color:#E0F7FA;
}
.sidebar-item:hover {
    background-color: rgba(255,255,255,0.1);
    border-radius:8px;
    transition: all 0.3s ease;
    padding-left: 6px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png", width=160)
st.sidebar.markdown("<div class='sidebar-title'>⚙️ KârMentor Menü</div>", unsafe_allow_html=True)
lang = st.sidebar.radio("🌍 Language / Dil", ["Türkçe", "English"], index=0 if st.session_state.lang == "Türkçe" else 1)
dark = st.sidebar.toggle("🌗 Dark Mode", value=st.session_state.dark)
st.session_state.lang = lang
st.session_state.dark = dark

# =========================================
# THEMES + GRADIENT BACKGROUND
# =========================================
if st.session_state.dark:
    bg_gradient = """
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
    """
    text_color = "#E0F7FA"
    card = "#132E35"
    accent = "#00B8BE"
else:
    bg_gradient = """
        background: linear-gradient(135deg, #1A3C40, #165E63, #1A3C40);
        background-size: 400% 400%;
        animation: gradientMove 20s ease infinite;
    """
    text_color = "#E6FAFA"
    card = "#244D52"
    accent = "#00C2CB"

st.markdown(f"""
<style>
@keyframes gradientMove {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}
html, body, [class*="stApp"], section.main {{
    {bg_gradient}
    color: {text_color} !important;
    font-family: 'Inter', sans-serif;
}}
.stButton>button {{
    background: {accent} !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
}}
.stButton>button:hover {{
    background: #00E0E7 !important;
    transform: scale(1.03);
}}
.stDataEditor, .stDataFrame {{
    background: {card} !important;
    border-radius: 10px;
    border: 1px solid {accent};
    color: {text_color} !important;
}}
div[data-testid="stTabs"] > div > div > button {{
    background-color: {card} !important;
    color: {text_color} !important;
    border-radius: 12px !important;
    border: 2px solid {accent} !important;
    font-size: 18px !important;
    padding: 10px 20px !important;
    margin-right: 8px !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)

# =========================================
# NAVIGATION TABS
# =========================================
tabs = st.tabs([
    "🏠 Ana Sayfa / Home",
    "📋 Veri / Data",
    "🧠 Model / Model",
    "💰 Tahmin / Prediction"
])

# =========================================
# HOME TAB
# =========================================
with tabs[0]:
    st.markdown("""
    <div style='text-align:center;'>
        <img src="https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png" width="220">
        <h2>💼 KârMentor - AI Profit Decision Assistant</h2>
        <p style='font-size:18px;'>Analyze your data, train your model, and predict profits with style 🚀</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# DATA TAB
# =========================================
with tabs[1]:
    st.subheader("📋 Veri Düzenleme")
    veri = st.data_editor(st.session_state.veri, num_rows="dynamic", use_container_width=True)
    st.session_state.veri = veri
    st.caption("Verileri düzenleyebilir veya yenilerini ekleyebilirsin.")

# =========================================
# MODEL TAB
# =========================================
with tabs[2]:
    st.subheader("🧠 Model Eğitimi")
    if len(st.session_state.veri) < 3:
        st.warning("Modeli eğitmek için en az 3 satır veri gereklidir.")
    else:
        if st.button("🧠 Modeli Eğit ve Aktifleştir"):
            with st.spinner("🔄 Model eğitiliyor..."):
                time.sleep(1)
                X = st.session_state.veri.drop(columns=["Kâr Oranı"])
                y = st.session_state.veri["Kâr Oranı"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=150, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = round(r2_score(y_test, preds), 3)
                mae = round(mean_absolute_error(y_test, preds), 3)
            st.success(f"✅ R²: {r2} | MAE: {mae}")
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/karmentor_model.joblib")
            st.session_state.model = model

# =========================================
# PREDICTION TAB + GRAPH
# =========================================
with tabs[3]:
    st.subheader("💰 Kâr Tahmini ve Görselleştirme")

    if not st.session_state.model:
        st.warning("Lütfen önce modeli eğitin.")
    else:
        fiyat = st.number_input("Birim Fiyat (₺)", value=120.0, step=10.0)
        maliyet = st.number_input("Birim Maliyet (₺)", value=70.0, step=5.0)
        satis = st.number_input("Satış Adedi", value=25, step=1)
        indirim = st.slider("İndirim Oranı (%)", 0.0, 0.5, 0.1)

        if st.button("📈 Kârı Hesapla"):
            yeni = pd.DataFrame([[fiyat, maliyet, satis, indirim]],
                                columns=["Birim Fiyat", "Birim Maliyet", "Satış Adedi", "İndirim Oranı"])
            oran = st.session_state.model.predict(yeni)[0]
            kar_tutar = (fiyat - maliyet) * satis * (1 - indirim)

            st.metric("💸 Tahmini Kâr Oranı (%)", f"{oran*100:.2f}")
            st.metric("💰 Tahmini Kâr Tutarı (₺)", f"{kar_tutar:,.2f}")

            fiyat_araligi = list(range(int(fiyat*0.7), int(fiyat*1.3), 5))
            tahminler = []
            for f in fiyat_araligi:
                yeni = pd.DataFrame([[f, maliyet, satis, indirim]],
                                    columns=["Birim Fiyat", "Birim Maliyet", "Satış Adedi", "İndirim Oranı"])
                tahminler.append(st.session_state.model.predict(yeni)[0]*100)

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(fiyat_araligi, tahminler, color=accent, linewidth=3, marker="o")
            ax.set_title("💹 Fiyat - Kâr Oranı Grafiği", color=text_color)
            ax.set_xlabel("Birim Fiyat (₺)", color=text_color)
            ax.set_ylabel("Kâr Oranı (%)", color=text_color)
            ax.tick_params(colors=text_color)
            fig.patch.set_facecolor("#0D1F21")
            ax.set_facecolor(card)
            st.pyplot(fig)

# =========================================
# FOOTER
# =========================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    "<p style='text-align:center;'>© 2025 KârMentor | Geliştiren: Harun Ağırman</p>",
    unsafe_allow_html=True
)
