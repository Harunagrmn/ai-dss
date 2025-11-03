import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

st.set_page_config(page_title="Kâr Hesaplama Asistanı", layout="wide")

st.title("💼 Yapay Zekâ Destekli Kâr Hesaplama Asistanı")
st.write("""
Bu sistem, girdiğiniz bilgilere göre **işletmenizin kâr oranını** tahmin eder.  
Kendi verilerinizi girip, olası senaryolara göre ne kadar kâr elde edeceğinizi görebilirsiniz.
""")

# ----------------------------
# 📊 1️⃣ Örnek Veriler
# ----------------------------
st.subheader("📋 Örnek Veriler (İstersen Değiştir)")

default_data = pd.DataFrame([
    [120, 35, 0.05, 60, 50, 7, 0.02, 5, 10, 0.38],
    [200, 18, 0.10, 45, 80, 9, 0.03, 6, 12, 0.33],
    [130, 42, 0.03, 70, 55, 6, 0.02, 7, 8, 0.44],
    [90,  60, 0.00, 90, 40, 8, 0.04, 5, 13, 0.45],
    [210, 25, 0.05, 55, 100,10, 0.03, 6, 9, 0.39]
], columns=[
    "Birim Fiyat", "Satış Adedi", "İndirim Oranı",
    "Stok Miktarı", "Birim Maliyet", "Tedarik Süresi (gün)",
    "Stok Tutma Oranı", "Satış Ekibi Sayısı", "Mesai (saat)", "Kâr Oranı"
])

if "veri" not in st.session_state:
    st.session_state.veri = default_data.copy()

veri = st.session_state.veri

st.data_editor(
    veri,
    num_rows="dynamic",
    use_container_width=True,
    key="veri_editor"
)

st.session_state.veri = veri
st.info("Tablodaki değerleri değiştirebilir veya yeni satırlar ekleyebilirsin.")

# ----------------------------
# 🧠 2️⃣ Model Eğitimi
# ----------------------------
st.subheader("🤖 Modeli Eğit")

if len(veri) < 5:
    st.warning("Modeli eğitmek için en az 5 satır veri gereklidir.")
else:
    if st.button("🔁 Modeli Eğit ve Kâr Tahmini Aktifleştir"):
        X = veri.drop(columns=["Kâr Oranı"])
        y = veri["Kâr Oranı"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        r2 = round(r2_score(y_test, preds), 3)
        mae = round(mean_absolute_error(y_test, preds), 3)

        st.success(f"Model başarıyla eğitildi ✅ | Doğruluk (R²): {r2} | Ortalama Hata: {mae}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/kazanc_modeli.joblib")
        st.session_state.model = model

# ----------------------------
# 💰 3️⃣ Kâr Tahmini
# ----------------------------
st.subheader("💰 Kâr Oranı Tahmini Yap")

if "model" in st.session_state:
    st.write("Yeni bir senaryo gir ve kâr oranını öğren:")

    col1, col2, col3 = st.columns(3)

    with col1:
        fiyat = st.number_input("Birim Fiyat (₺)", value=150.0, step=10.0)
        satis = st.number_input("Satış Adedi", value=30.0, step=1.0)
        indirim = st.slider("İndirim Oranı (%)", 0.0, 0.5, 0.1)

    with col2:
        stok = st.number_input("Stok Miktarı", value=50.0, step=5.0)
        maliyet = st.number_input("Birim Maliyet (₺)", value=60.0, step=5.0)
        tedarik = st.number_input("Tedarik Süresi (gün)", value=7.0, step=1.0)

    with col3:
        stok_maliyeti = st.slider("Stok Tutma Oranı (%)", 0.0, 0.1, 0.02)
        ekip = st.number_input("Satış Ekibi Sayısı", value=5.0, step=1.0)
        mesai = st.number_input("Mesai (saat)", value=10.0, step=1.0)

    input_df = pd.DataFrame([[
        fiyat, satis, indirim, stok, maliyet,
        tedarik, stok_maliyeti, ekip, mesai
    ]], columns=[
        "Birim Fiyat", "Satış Adedi", "İndirim Oranı",
        "Stok Miktarı", "Birim Maliyet", "Tedarik Süresi (gün)",
        "Stok Tutma Oranı", "Satış Ekibi Sayısı", "Mesai (saat)"
    ])

    if st.button("📈 Kâr Tahminini Hesapla"):
        tahmin = st.session_state.model.predict(input_df)[0]
        st.success(f"Tahmini Kâr Oranı: **%{tahmin * 100:.2f}**")
        st.write("---")
        st.write("📊 Girdiğiniz senaryo:")
        st.dataframe(input_df)
else:
    st.warning("Model henüz eğitilmedi. Lütfen önce eğitimi başlat.")

st.caption("© 2025 Kâr Hesaplama Asistanı | Geliştiren: Harun Ağırman")
