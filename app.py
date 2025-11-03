import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import time

# =============================
# PAGE SETTINGS
# =============================
st.set_page_config(page_title="KârMentor | AI Profit Assistant", layout="wide", page_icon="💼")

# =============================
# SESSION VARS (Initialize)
# =============================
defaults = {
    "logged_in": False,
    "lang": "Türkçe",
    "dark": False,
    "model": None,
    "veri": pd.DataFrame([
        [100, 60, 20, 0.10, 0.40],
        [120, 70, 25, 0.05, 0.42],
        [150, 80, 30, 0.00, 0.50],
        [90,  55, 18, 0.15, 0.33],
        [130, 75, 28, 0.07, 0.46],
    ], columns=["Birim Fiyat", "Birim Maliyet", "Satış Adedi", "İndirim Oranı", "Kâr Oranı"])
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================
# DARK MODE & LANGUAGE THEME
# =============================
if st.session_state.dark:
    bg = "#0F2027"
    card = "#132E35"
    fg = "#E0F7FA"
    accent = "#00B8BE"
else:
    bg = "#1A3C40"
    card = "#244D52"
    fg = "#E6FAFA"
    accent = "#00C2CB"

# =============================
# STYLE (Gradient + Theme)
# =============================
st.markdown(f"""
<style>
@keyframes gradientMove {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}
html, body, [class*="stApp"], section.main {{
    background: linear-gradient(135deg, {bg}, #165E63, {bg});
    background-size: 400% 400%;
    animation: gradientMove 20s ease infinite;
    color: {fg} !important;
    font-family: 'Inter', sans-serif;
}}
h1,h2,h3,h4,h5,h6,p,span,label {{
    color: {fg} !important;
}}
.stButton>button {{
    background: {accent} !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease-in-out;
}}
.stButton>button:hover {{
    background: #02D6DD !important;
    transform: scale(1.03);
}}
.stTextInput>div>div>input {{
    background-color: {card} !important;
    color: {fg} !important;
    border-radius: 6px;
}}
</style>
""", unsafe_allow_html=True)

# =============================
# LOGIN SCREEN
# =============================
if not st.session_state.logged_in:
    st.markdown("<h2 style='text-align:center;'>💼 KârMentor Giriş</h2>", unsafe_allow_html=True)
    st.image("https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png", width=200)
    st.write("")
    username = st.text_input("👤 Kullanıcı Adı", placeholder="örnek: Harunagrmn")
    password = st.text_input("🔑 Şifre", type="password", placeholder="********")
    st.write("")
    if st.button("🚀 Giriş Yap"):
        if username.strip() != "" and password.strip() != "":
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Hoş geldin, {username}! 🎉")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Lütfen kullanıcı adı ve şifre gir.")
    st.stop()

# =============================
# SIDEBAR SETTINGS
# =============================
st.sidebar.image("https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png", width=160)
st.sidebar.markdown(f"<h4 style='text-align:center;'>👋 Hoş geldin, {st.session_state.username}</h4>", unsafe_allow_html=True)

lang_choice = st.sidebar.radio("🌍 Dil Seç / Language", ["Türkçe", "English"], 
                               index=0 if st.session_state.lang == "Türkçe" else 1)
if lang_choice != st.session_state.lang:
    st.session_state.lang = lang_choice
    st.rerun()

dark_toggle = st.sidebar.toggle("🌗 Dark Mode", value=st.session_state.dark)
if dark_toggle != st.session_state.dark:
    st.session_state.dark = dark_toggle
    st.rerun()

logout = st.sidebar.button("🚪 Çıkış Yap")
if logout:
    st.session_state.logged_in = False
    st.rerun()

# =============================
# TAB STRUCTURE
# =============================
tabs = st.tabs(["🏠 Ana Sayfa", "📋 Veri", "🧠 Model", "💰 Tahmin"])

# =============================
# HOME TAB
# =============================
with tabs[0]:
    st.markdown(f"""
    <div style='text-align:center;'>
        <img src="https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png" width="220">
        <h2>💼 KârMentor - Akıllı Kâr Asistanı</h2>
        <p>{'Fiyat, maliyet, satış ve indirim oranlarına göre tahmini kâr hesaplamaları yapar ve öneriler sunar.' if st.session_state.lang == 'Türkçe' else 'Analyze pricing, costs and sales data to forecast profits and insights.'}</p>
    </div>
    """, unsafe_allow_html=True)

# =============================
# DATA TAB
# =============================
with tabs[1]:
    st.subheader("📋 Veri Düzenleme" if st.session_state.lang == "Türkçe" else "📋 Data Editor")
    veri = st.data_editor(st.session_state.veri, num_rows="dynamic", use_container_width=True)
    st.session_state.veri = veri
    st.caption("Verileri düzenleyebilir veya yenilerini ekleyebilirsin." if st.session_state.lang == "Türkçe"
               else "You can edit or add new rows to the dataset.")

# =============================
# MODEL TAB
# =============================
with tabs[2]:
    st.subheader("🧠 Model Eğitimi" if st.session_state.lang == "Türkçe" else "🧠 Model Training")
    if len(st.session_state.veri) < 3:
        st.warning("Modeli eğitmek için en az 3 satır veri gereklidir." if st.session_state.lang == "Türkçe"
                   else "At least 3 rows are required to train the model.")
    else:
        if st.button("🧠 Modeli Eğit ve Aktifleştir" if st.session_state.lang == "Türkçe" else "🧠 Train Model"):
            with st.spinner("🔄 Model eğitiliyor..." if st.session_state.lang == "Türkçe" else "🔄 Training model..."):
                X = st.session_state.veri.drop(columns=["Kâr Oranı"])
                y = st.session_state.veri["Kâr Oranı"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=150, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = round(r2_score(y_test, preds), 3)
                mae = round(mean_absolute_error(y_test, preds), 3)
            st.success(f"✅ Model Eğitildi | R²: {r2} | Hata: {mae}" if st.session_state.lang == "Türkçe"
                       else f"✅ Model Trained | R²: {r2} | MAE: {mae}")
            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/karmentor_model.joblib")
            st.session_state.model = model

# =============================
# PREDICTION TAB
# =============================
with tabs[3]:
    st.subheader("💰 Kâr Tahmini" if st.session_state.lang == "Türkçe" else "💰 Profit Prediction")
    if not st.session_state.model:
        st.warning("Lütfen önce modeli eğitin." if st.session_state.lang == "Türkçe"
                   else "Please train the model first.")
    else:
        fiyat = st.number_input("Birim Fiyat (₺)" if st.session_state.lang == "Türkçe" else "Unit Price (₺)", value=120.0)
        maliyet = st.number_input("Birim Maliyet (₺)" if st.session_state.lang == "Türkçe" else "Unit Cost (₺)", value=70.0)
        satis = st.number_input("Satış Adedi" if st.session_state.lang == "Türkçe" else "Sales Quantity", value=25)
        indirim = st.slider("İndirim Oranı (%)" if st.session_state.lang == "Türkçe" else "Discount Rate (%)", 0.0, 0.5, 0.1)

        if st.button("📈 Kârı Hesapla" if st.session_state.lang == "Türkçe" else "📈 Calculate Profit"):
            yeni = pd.DataFrame([[fiyat, maliyet, satis, indirim]],
                                columns=["Birim Fiyat", "Birim Maliyet", "Satış Adedi", "İndirim Oranı"])
            oran = st.session_state.model.predict(yeni)[0]
            kar_tutar = (fiyat - maliyet) * satis * (1 - indirim)

            st.metric("💸 Tahmini Kâr Oranı (%)" if st.session_state.lang == "Türkçe" else "💸 Predicted Profit Margin (%)",
                      f"{oran*100:.2f}")
            st.metric("💰 Tahmini Kâr Tutarı (₺)" if st.session_state.lang == "Türkçe" else "💰 Predicted Profit (₺)",
                      f"{kar_tutar:,.2f}")

# =============================
# FOOTER
# =============================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    f"<p style='text-align:center;'>© 2025 KârMentor | {'Geliştiren' if st.session_state.lang == 'Türkçe' else 'Developed by'} Harun Ağırman</p>",
    unsafe_allow_html=True
)
