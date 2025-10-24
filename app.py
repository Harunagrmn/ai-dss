import streamlit as st
import pandas as pd
import joblib
from src.features import load_data
from src.utils import optimize_profit
from src.reporting import build_report_bytes

# ==============================
# 🧩 1. SAYFA AYARLARI
# ==============================
st.set_page_config(page_title="Yapay Zekâ Destekli Karar Destek Sistemi", layout="wide")

st.title("🧠 Yapay Zekâ Destekli Karar Destek Sistemi")
st.write("Bu sistem, satış, stok ve personel verilerini kullanarak kâr marjını tahmin eder ve PDF raporu oluşturur.")

# ==============================
# 🔐 2. LOGIN & ROLLER
# ==============================

USERS = {
    "admin":   {"password": "admin123",   "role": "admin"},
    "analyst": {"password": "analyst123", "role": "analyst"},
    "manager": {"password": "manager123", "role": "manager"},
}

if "auth" not in st.session_state:
    st.session_state.auth = False
    st.session_state.role = None
    st.session_state.user = None

def login_ui():
    st.header("🔐 Giriş Yap")
    username = st.text_input("Kullanıcı Adı")
    password = st.text_input("Şifre", type="password")
    if st.button("Giriş"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state.auth = True
            st.session_state.role = USERS[username]["role"]
            st.session_state.user = username
            st.success(f"Hoş geldin, {username} ({st.session_state.role})")
            st.rerun()
        else:
            st.error("❌ Hatalı kullanıcı adı veya şifre!")

if not st.session_state.auth:
    login_ui()
    st.stop()

# Kenar menü
st.sidebar.success(f"👤 {st.session_state.user} ({st.session_state.role})")
if st.sidebar.button("🚪 Çıkış Yap"):
    for k in ["auth", "role", "user"]:
        st.session_state.pop(k, None)
    st.rerun()

# ==============================
# 📂 3. YETKİLERE GÖRE ÖZELLİKLER
# ==============================

if st.session_state.role == "admin":
    st.sidebar.header("🗂️ Veri Yükleme (Admin)")
    sales_file = st.sidebar.file_uploader("sales.csv", type="csv")
    inventory_file = st.sidebar.file_uploader("inventory.csv", type="csv")
    staff_file = st.sidebar.file_uploader("staff.csv", type="csv")
    if st.sidebar.button("Verileri Kaydet"):
        import pandas as pd
        if sales_file: pd.read_csv(sales_file).to_csv("data/sales.csv", index=False)
        if inventory_file: pd.read_csv(inventory_file).to_csv("data/inventory.csv", index=False)
        if staff_file: pd.read_csv(staff_file).to_csv("data/staff.csv", index=False)
        st.sidebar.success("✅ Veriler başarıyla kaydedildi!")

if st.session_state.role in ["analyst", "admin"]:
    st.sidebar.header("🧠 Model Eğitimi")
    if st.sidebar.button("🔁 Modeli Yeniden Eğit"):
        from src.train import train_model
        train_model()
        st.sidebar.success("✅ Model yeniden eğitildi!")

# ==============================
# 📊 4. KARAR DESTEK ANA EKRANI
# ==============================

if st.session_state.role not in ["manager", "analyst", "admin"]:
    st.warning("Bu bölümü görüntüleme yetkiniz yok.")
    st.stop()

# Modeli yükle
@st.cache_resource
def load_model():
    model = joblib.load("models/profit_model.joblib")
    return model

model = load_model()
df = load_data()

defaults = {
    "unit_price": float(df["unit_price"].mean()),
    "units_sold": float(df["units_sold"].mean()),
    "discount_rate": float(df["discount_rate"].mean()),
    "stock_qty": float(df["stock_qty"].mean()),
    "unit_cost": float(df["unit_cost"].mean()),
    "lead_time_days": float(df["lead_time_days"].mean()),
    "holding_cost_rate": float(df["holding_cost_rate"].mean()),
    "sales_team_size": float(df["sales_team_size"].mean()),
    "overtime_hours": float(df["overtime_hours"].mean()),
}

st.subheader("🔧 Senaryo Girdileri")

col1, col2, col3 = st.columns(3)
with col1:
    unit_price = st.number_input("Birim Fiyat", value=defaults["unit_price"])
    units_sold = st.number_input("Satış Adedi", value=defaults["units_sold"])
    discount_rate = st.slider("İndirim Oranı", 0.0, 0.5, float(defaults["discount_rate"]), 0.01)

with col2:
    stock_qty = st.number_input("Stok Miktarı", value=defaults["stock_qty"])
    unit_cost = st.number_input("Birim Maliyet", value=defaults["unit_cost"])
    lead_time_days = st.number_input("Tedarik Süresi (gün)", value=defaults["lead_time_days"])

with col3:
    holding_cost_rate = st.slider("Stok Tutma Oranı", 0.0, 0.1, float(defaults["holding_cost_rate"]), 0.005)
    sales_team_size = st.number_input("Satış Ekibi Sayısı", value=defaults["sales_team_size"])
    overtime_hours = st.number_input("Mesai (Saat)", value=defaults["overtime_hours"])

# ------------------------------
# 📈 Tahmin & PDF Raporu
# ------------------------------
if st.button("📈 Tahmini Kâr Marjını Hesapla"):
    X = pd.DataFrame([[
        unit_price, units_sold, discount_rate, stock_qty, unit_cost,
        lead_time_days, holding_cost_rate, sales_team_size, overtime_hours
    ]], columns=[
        "unit_price", "units_sold", "discount_rate", "stock_qty", "unit_cost",
        "lead_time_days", "holding_cost_rate", "sales_team_size", "overtime_hours"
    ])

    prediction = model.predict(X)[0]
    st.success(f"Tahmini Kâr Marjı: **{prediction:.2%}**")

    scenario = {
        "unit_price": unit_price,
        "units_sold": units_sold,
        "discount_rate": discount_rate,
        "stock_qty": stock_qty,
        "unit_cost": unit_cost,
        "lead_time_days": lead_time_days,
        "holding_cost_rate": holding_cost_rate,
        "sales_team_size": sales_team_size,
        "overtime_hours": overtime_hours
    }

    results = {"Tahmini Kâr Marjı": f"{prediction:.2%}"}
    pdf_bytes = build_report_bytes("Karar Raporu", scenario, results)
    st.download_button("📄 PDF Raporu İndir", data=pdf_bytes, file_name="karar_raporu.pdf", mime="application/pdf")

st.write("---")

# ------------------------------
# 🔮 En Kârlı Senaryo
# ------------------------------
st.subheader("🔮 En Kârlı Senaryoyu Bul")
if st.button("⚡ Optimize Et"):
    base_inputs = {
        "unit_price": unit_price,
        "units_sold": units_sold,
        "discount_rate": discount_rate,
        "stock_qty": stock_qty,
        "unit_cost": unit_cost,
        "lead_time_days": lead_time_days,
        "holding_cost_rate": holding_cost_rate,
        "sales_team_size": sales_team_size,
        "overtime_hours": overtime_hours
    }

    best = optimize_profit(model, base_inputs)
    if best:
        st.success(
            f"""
            **💡 En Kârlı Senaryo:**
            \n💰 Tahmini Kâr Marjı: **{best['predicted_profit']}%**
            \n📈 Fiyat Değişimi: {best['unit_price_change']}
            \n🏷️ İndirim Değişimi: {best['discount_change']}
            \n📦 Stok Değişimi: {best['stock_change']}
            """
        )

        opt_results = {
            "En İyi Tahmini Kâr Marjı": f"{best['predicted_profit']}%",
            "Fiyat Değişimi": best["unit_price_change"],
            "İndirim Değişimi": best["discount_change"],
            "Stok Değişimi": best["stock_change"]
        }

        opt_pdf = build_report_bytes("Optimize Senaryo Raporu", base_inputs, opt_results)
        st.download_button("📄 Optimize Raporu İndir", data=opt_pdf, file_name="optimize_raporu.pdf", mime="application/pdf")

st.write("---")
st.caption("© 2025 Yapay Zekâ Destekli Karar Destek Sistemi - Muhammed Harun Ağırman")
