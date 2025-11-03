# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np # Optimizasyon icin eklendi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import time

# OpenAI kütüphanesini kullaniyoruz (OpenRouter icin)
from openai import OpenAI

# =========================================
# DIL DESTEGI (TÜM PROMPT'LAR DÜZELTİLDİ)
# =========================================
translations = {
    "tr": {
        "page_title": "KarMentor | AI Kâr Asistanı",
        "sidebar_title": "⚙️ KarMentor Menü",
        "lang_select_label": "🌍 Language / Dil",
        "lang_options": ["Turkce", "English"],
        "dark_mode_label": "🌗 Karanlık Mod",
        "chatbot_popover_label": "🤖 Haruncuk'a Soru Sor",
        "chatbot_input_placeholder": "Bana bir soru sor...",
        "chatbot_thinking": "Haruncuk düşünüyor...",
        "chatbot_error": "Haruncuk bir hatayla karşılaştı:",
        "chatbot_clear_button": "Sohbeti Temizle",
        
        # !!! 1. DÜZELTME BURADA (HARUNCUK'UN BEYNİ) !!!
        # Daha net ve kati bir görev tanimi.
        "chatbot_system_prompt": """Sen, KarMentor uygulamasinin yardimci botu Haruncuk'sun.
Senin TEK GOREVIN, kullanicinin bu uygulama hakkindaki sorularini yanitlamaktir.
Bu uygulama 3 adimda calisir:
1. VERI YUKLEME: Kullanici 'Birim Fiyat', 'Birim Maliyet', 'Indirim Orani' ve 'Satis Adedi' kolonlarini iceren kendi .csv veya .xlsx dosyasini yukler. Veya ornek veriyi kullanir.
2. MODEL EGITIMI: Kullanici, yukledigi bu veriyi kullanarak 'Satis Adedi'ni (talebi) tahmin eden bir yapay zeka modeli egitir.
3. TAHMIN VE OPTIMIZASYON: Kullanici, egitilmis bu modeli kullanarak, farkli fiyat senaryolarinda tahmini satis adedini ve karini gorur. 'En Iyi Fiyati Bul' araci ise en yuksek kari getirecek fiyati simule eder.
Disaridan baska bir bilgin yok. 'Bu uygulama ne ise yarar?' diye sorarlarsa, bu 3 adimi ozetle.""",
        
        "tabs": ["🏠 Ana Sayfa", "📋 Veri", "🧠 Model", "💰 Tahmin"],
        "home_title": "💼 KarMentor - Yapay Zeka Kâr Destek Sistemi",
        "home_subtitle": "Verilerinizi analiz edin, modelinizi eğitin ve kârlılığınızı optimize edin 🚀",
        
        "home_guide_header": "Uygulama Nasıl Çalışır?",
        "home_guide_1_header": "1. Veri Yükleme",
        "home_guide_1_text": "KarMentor, sizin kendi satış verilerinizle çalışır. **'Veri'** sekmesine giderek, gerekli kolonları ('Birim Fiyat', 'Birim Maliyet', 'İndirim Oranı', 'Satış Adedi') içeren .csv veya .xlsx dosyanızı yükleyin. Eğer denemek için bir dosyanız yoksa, **'Örnek Verilerle Deneyin'** butonunu kullanabilirsiniz.",
        "home_guide_2_header": "2. Model Eğitimi",
        "home_guide_2_text": "Verinizi yükledikten sonra **'Model'** sekmesine geçin. Buradaki butona tıklayarak, verileriniz arasındaki gizli ilişkileri (örneğin, fiyatın satış miktarına etkisini) öğrenen yapay zeka modelinizi saniyeler içinde eğitin. Bu model, uygulamanın 'beyni' olacaktır.",
        "home_guide_3_header": "3. Tahmin ve Optimizasyon",
        "home_guide_3_text": "Modeliniz hazır olduğunda, **'Tahmin'** sekmesine gidin. Burada iki güçlü araç bulacaksınız: **Tekli Tahmin** (belirli bir fiyata karşılık tahmini satış ve kârı gösterir) ve **Kâr Optimizasyonu** (belirlediğiniz bir fiyat aralığını tarayarak size en yüksek kârı getirecek 'altın' fiyatı bulur).",
        "home_guide_4_header": "Yardıma mı İhtiyacınız Var? 🤖",
        "home_guide_4_text": "Herhangi bir adımda takılırsanız veya 'Birim Fiyat nedir?' gibi bir sorunuz olursa, sağ alttaki **'Haruncuk'a Soru Sor'** butonunu kullanmaktan çekinmeyin!",

        "data_header": "📋 Veri Yükleme",
        "data_upload_label": "Lütfen satış verilerinizi (.csv veya .xlsx) buraya yükleyin:",
        "data_upload_help": "Dosyanızda 'Birim Fiyat', 'Birim Maliyet', 'İndirim Oranı' ve 'Satış Adedi' kolonları olmalıdır. (Kâr Oranı artık gerekli değil).",
        "data_sample_button": "Örnek Verilerle Deneyin",
        "data_sample_loading": "Örnek veri yükleniyor...",
        "data_upload_success": "✅ Veri başarıyla yüklendi! Toplam {rows} satır.",
        "data_upload_info": "Lütfen 'Model' sekmesine geçmeden önce verilerinizi yükleyin veya örnek verileri kullanın.",
        "data_upload_error": "❌ Dosya okunamadı. Lütfen geçerli bir .csv veya .xlsx dosyası olduğundan emin olun.",
        "data_preview_header": "Veri Önizlemesi (İlk 10 Satır)",

        "model_header": "🧠 Model Eğitimi (Satış Adedi Tahmini)",
        "model_warning_upload": "Lütfen önce 'Veri' sekmesinden kendi verilerinizi yükleyin veya örnek verileri kullanın.",
        "model_button": "🧠 Modeli YÜKLEDİĞİM VERİ ile Eğit ve Aktifleştir",
        "model_spinner": "🔄 Model eğitiliyor (Hedef: Satış Adedi)...",
        "model_success_metrics": "✅ R²: {r2} | MAE: {mae} (Satış Adedi)",
        "model_success_no_test": "✅ Model eğitildi (Test verisi yetersiz, tüm veri kullanıldı).",
        
        "predict_header": "💰 Kâr Tahmini ve Optimizasyonu",
        "predict_warning_model": "Lütfen önce 'Model' sekmesinden Satış Adedi modelini eğitin.",
        "predict_warning_api": "Yapay Zeka (OpenRouter) API anahtarı başlatılamadı. Lütfen üstteki hata mesajını ve secrets dosyanızı kontrol edin.",
        "predict_input_price": "Birim Fiyat (TL)",
        "predict_input_cost": "Birim Maliyet (TL)",
        "predict_input_discount": "İndirim Oranı",
        "predict_button_calculate": "📈 Kârı Hesapla (Satış Adedini Tahmin Et)",
        "predict_metric_profit_rate": "💸 Tahmini Satış Adedi",
        "predict_metric_profit_amount": "💰 Tahmini Kâr Tutarı (TL)",
        "predict_plot_title": "💹 Fiyat - Tahmini Satış Adedi Grafiği",
        "predict_plot_xlabel": "Birim Fiyat (TL)",
        "predict_plot_ylabel": "Tahmini Satış Adedi",
        
        # !!! DEGISIKLIK BURADA !!!
        "predict_analysis_header": "🤖 Haruncuk Yapay Zeka Analizi", 
        "predict_analysis_button": "💡 Bu Senaryo Icin Analiz Al",
        "predict_analysis_spinner": "🔄 Haruncuk bu senaryoyu analiz ediyor...",
        "predict_analysis_error": "Haruncuk analiz yaparken bir hatayla karşılaştı:",
        
        # !!! 2. DÜZELTME BURADA (ANALIZ BUTONU BEYNI - SISTEM) !!!
        "predict_analysis_prompt_system": """Sen KarMentor adlı bir iş zekası uygulamasının finansal analiz asistanısın.
Rolün, bir makine öğrenimi modelinin (RandomForest) tahminini alıp,
bunu CEO'larin anlayacagi dilde, eyleme gecirilebilir bir is tavsiyesine donusturmektir.
Cevabin sadece analizin kendisi olsun, "Elbet ki, iste analiz:" gibi baslama.""",
        
        # !!! 3. DÜZELTME BURADA (ANALIZ BUTONU BEYNI - KULLANICI) !!!
        "predict_analysis_prompt_user": """
Asagidaki senaryoyu analiz et:
Girdiler:
- Birim Fiyat (TL): {fiyat}
- Birim Maliyet (TL): {maliyet}
- Indirim Orani: {indirim}%
Model Tahmini (Bu girdilere gore):
- Tahmini Satis Adedi: {satis} adet
Hesaplanan Sonuc:
- Tahmini Toplam Kar (TL): {tutar}
Lutfen bu senaryoya dayanarak kisa (3-4 cumlelik) bir analiz ve stratejik bir oneri sun.
(Fiyatin satis adedini nasil etkiledigine deginebilirsin).
""",
        
        "optim_header": "🎯 Kâr Optimizasyonu (Fiyata Göre)",
        "optim_help": "Maliyet ve İndirim sabitken, modelin satış adedi tahminlerini kullanarak en yüksek Kâr Tutarını hangi fiyatın getireceğini bulun.",
        "optim_min_price": "Minimum Fiyat (TL)",
        "optim_max_price": "Maksimum Fiyat (TL)",
        "optim_button": "🚀 En İyi Fiyatı Bul (Akıllı Simülasyon)",
        "optim_spinner": "🔄 Fiyat aralığı taranarak optimizasyon yapılıyor (Model Satış Adedini Tahmin Ediyor)...",
        "optim_success_title": "🏆 Optimizasyon Tamamlandı!",
        "optim_success_metric_price": "En Kârlı Fiyat (TL)",
        "optim_success_metric_profit": "Maksimum Tahmini Kâr (TL)",
        "optim_success_metric_sales": "O Fiyattaki Tahmini Satış Adedi",
        
        "footer_text": "© 2025 KarMentor | Geliştiren: Harun Ağırman"
    },
    "en": {
        "page_title": "KarMentor | AI Profit Assistant",
        "sidebar_title": "⚙️ KarMentor Menu",
        "lang_select_label": "🌍 Language / Dil",
        "lang_options": ["Turkce", "English"],
        "dark_mode_label": "🌗 Dark Mode",
        "chatbot_popover_label": "🤖 Ask Haruncuk",
        "chatbot_input_placeholder": "Ask me a question...",
        "chatbot_thinking": "Haruncuk is thinking...",
        "chatbot_error": "Haruncuk encountered an error:",
        "chatbot_clear_button": "Clear Chat",
        
        "chatbot_system_prompt": """You are Haruncuk, the assistant bot for the KarMentor app.
Your ONLY JOB is to answer questions about this app.
This app works in 3 steps:
1. DATA UPLOAD: The user uploads their own .csv or .xlsx file with 'Birim Fiyat', 'Birim Maliyet', 'Indirim Orani', and 'Satis Adedi' columns. Or they use the sample data.
2. MODEL TRAINING: The user trains an AI model to predict 'Sales Quantity' (demand) based on their data.
3. PREDICTION & OPTIMIZATION: The user uses this trained model to predict sales/profit for different prices, and uses the 'Find Best Price' tool to simulate the most profitable price.
You have NO other knowledge. If they ask 'What is this app?', summarize these 3 steps.""",
        
        "tabs": ["🏠 Home", "📋 Data", "🧠 Model", "💰 Prediction"],
        "home_title": "💼 KarMentor - AI Profit Decision Assistant",
        "home_subtitle": "Analyze your data, train your model, and optimize your profitability 🚀",
        
        "home_guide_header": "How to Use This App?",
        "home_guide_1_header": "1. Upload Data",
        "home_guide_1_text": "KarMentor works with your own sales data. Go to the **'Data' tab** to upload your .csv or .xlsx file. Your file must include 'Birim Fiyat', 'Birim Maliyet', 'Indirim Orani', and 'Satis Adedi'. If you don't have a file, just click the **'Try with Sample Data'** button.",
        "home_guide_2_header": "2. Train Model",
        "home_guide_2_text": "Once your data is loaded, go to the **'Model' tab**. Click the button to train your custom AI model. This model learns the unique patterns in your data, like how price affects the quantity of sales. This is the 'brain' of the app.",
        "home_guide_3_header": "3. Predict & Optimize",
        "home_guide_3_text": "On the **'Prediction' tab**, you can use your trained model. The **Single Prediction** tool estimates sales and profit for a specific price. The **Profit Optimization** tool simulates thousands of scenarios to find the 'golden price' that will maximize your total profit.",
        "home_guide_4_header": "Need Help? 🤖",
        "home_guide_4_text": "If you get stuck or have a question, use the **'Ask Haruncuk'** button in the bottom-right corner!",

        "data_header": "📋 Data Upload",
        "data_upload_label": "Please upload your sales data (.csv or .xlsx) here:",
        "data_upload_help": "Your file must contain 'Birim Fiyat', 'Birim Maliyet', 'Indirim Orani', and 'Satis Adedi'. (Profit Rate is no longer needed).",
        "data_sample_button": "Try with Sample Data",
        "data_sample_loading": "Loading sample data...",
        "data_upload_success": "✅ Data loaded successfully! Total {rows} rows.",
        "data_upload_info": "Please upload your data or use the sample data to proceed.",
        "data_upload_error": "❌ Could not read file. Please ensure it is a valid .csv or .xlsx file.",
        "data_preview_header": "Data Preview (First 10 Rows)",

        "model_header": "🧠 Model Training (Predicting Sales Quantity)",
        "model_warning_upload": "Please upload or load sample data on the 'Data' tab first.",
        "model_button": "🧠 Train and Activate Model with MY DATA",
        "model_spinner": "🔄 Model is training (Target: Sales Quantity)...",
        "model_success_metrics": "✅ R²: {r2} | MAE: {mae} (Sales Quantity)",
        "model_success_no_test": "✅ Model trained (Insufficient test data, all data was used).",
        
        "predict_header": "💰 Profit Prediction and Optimization",
        "predict_warning_model": "Please train the Sales Quantity model on the 'Model' tab first.",
        "predict_warning_api": "AI (OpenRouter) API key could not be initialized. Please check your secrets file.",
        "predict_input_price": "Unit Price (TL)",
        "predict_input_cost": "Unit Cost (TL)",
        "predict_input_discount": "Discount Rate",
        "predict_button_calculate": "📈 Calculate Profit (Predict Sales Quantity)",
        "predict_metric_profit_rate": "💸 Estimated Sales Quantity",
        "predict_metric_profit_amount": "💰 Estimated Total Profit (TL)",
        "predict_plot_title": "💹 Price - Estimated Sales Quantity Graph",
        "predict_plot_xlabel": "Unit Price (TL)",
        "predict_plot_ylabel": "Estimated Sales Quantity",
        
        # !!! DEGISIKLIK BURADA !!!
        "predict_analysis_header": "🤖 Haruncuk AI Analysis",
        "predict_analysis_button": "💡 Get Analysis for This Scenario",
        "predict_analysis_spinner": "🔄 Haruncuk is analyzing this scenario...",
        "predict_analysis_error": "Haruncuk encountered an error while analyzing:",
        
        "predict_analysis_prompt_system": """You are a financial analysis assistant for an app called KarMentor.
Your role is to take the output from a machine learning model (RandomForest) and
translate it into an actionable business recommendation in plain language for a CEO.
Your response should be only the analysis itself, do not start with "Certainly, here is the analysis:".
""",
        
        "predict_analysis_prompt_user": """
Analyze the following scenario:
Inputs:
- Unit Price (TL): {fiyat}
- Unit Cost (TL): {maliyet}
- Discount Rate: {indirim}%
Model Prediction (for these inputs):
- Estimated Sales Quantity: {satis} units
Calculated Result:
- Estimated Total Profit (TL): {tutar}
Please provide a brief (3-4 sentences) analysis and one strategic recommendation.
(You can mention how price impacts the predicted sales quantity).
""",
        
        "optim_header": "🎯 Profit Optimization (by Price)",
        "optim_help": "Find the price that yields the highest Total Profit, using the model's sales quantity predictions.",
        "optim_min_price": "Minimum Price (TL)",
        "optim_max_price": "Maximum Price (TL)",
        "optim_button": "🚀 Find Best Price (Smart Simulation)",
        "optim_spinner": "🔄 Optimizing by scanning price range (Model is predicting sales)...",
        "optim_success_title": "🏆 Optimization Complete!",
        "optim_success_metric_price": "Most Profitable Price (TL)",
        "optim_success_metric_profit": "Maximum Estimated Profit (TL)",
        "optim_success_metric_sales": "Est. Sales at that Price",

        "footer_text": "© 2025 KarMentor | Developed by: Harun Agirman"
    }
}

# =========================================
# HARUNCUK AVATAR URL'SI
# =========================================
HARUNCUK_AVATAR_URL = "https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/haruncukbot.png"

# =========================================
# YENI: ORNEK VERI FONKSIYONU
# =========================================
def create_sample_data():
    """Kullanicinin denemesi icin bir ornek DataFrame olusturur."""
    np.random.seed(42)
    fiyatlar = np.random.randint(50, 200, 50)
    maliyetler = fiyatlar * np.random.uniform(0.4, 0.7, 50) 
    indirimler = np.random.choice([0.0, 0.1, 0.15, 0.2], 50)
    temel_satis = 100 - (fiyatlar / 3) 
    gurultu = np.random.randint(-10, 10, 50)
    satis_adedi = (temel_satis + gurultu - (indirimler * 50)).astype(int)
    satis_adedi = np.maximum(5, satis_adedi) 
    
    data = pd.DataFrame({
        "Birim Fiyat": fiyatlar,
        "Birim Maliyet": maliyetler,
        "Indirim Orani": indirimler,
        "Satis Adedi": satis_adedi
    })
    return data

# =========================================
# PAGE SETTINGS
# =========================================
if "lang" not in st.session_state:
    st.session_state.lang = "tr" 
t = translations[st.session_state.lang]

st.set_page_config(
    page_title=t["page_title"],
    layout="wide",
    page_icon="💼" 
)

# =========================================
# OPENROUTER (GPT) AYARI
# =========================================
try:
    if "OPENROUTER_API_KEY" not in st.secrets or not st.secrets["OPENROUTER_API_KEY"]:
        raise Exception("OPENROUTER_API_KEY .streamlit/secrets.toml dosyasinda bulunamadi.")
    
    api_key = st.secrets["OPENROUTER_API_KEY"]
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    AI_AKTIF = True
except Exception as e:
    AI_AKTIF = False
    st.error(f"""
    **Yapay Zeka (OpenRouter) Baslatilamadi!**
    Hata: {e}
    """)

# =========================================
# SESSION VARS
# =========================================
if "dark" not in st.session_state:
    st.session_state.dark = False
if "model" not in st.session_state:
    st.session_state.model = None
if "user_data" not in st.session_state:
    st.session_state.user_data = None 
if "haruncuk_messages" not in st.session_state:
    st.session_state.haruncuk_messages = []
    
FEATURES = ["Birim Fiyat", "Birim Maliyet", "Indirim Orani"]
TARGET = "Satis Adedi"
REQUIRED_COLS = FEATURES + [TARGET]

# =========================================
# SIDEBAR
# =========================================
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #012C2E, #024C4F, #026D71);
    animation: gradientShift 10s ease infinite;
    background-size: 400% 400%;
    color: white;
}
@keyframes gradientShift { 0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;} }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: white !important; }
.sidebar-title { text-align:center; font-size:22px; font-weight:bold; color:#E0F7FA; }
.sidebar-item:hover { background-color: rgba(255,255,255,0.1); border-radius:8px; transition: all 0.3s ease; padding-left: 6px; }
</style>
""", unsafe_allow_html=True)
st.sidebar.image("https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png", width=160)
st.sidebar.markdown(f"<div class='sidebar-title'>{t['sidebar_title']}</div>", unsafe_allow_html=True)
lang_index = 0 if st.session_state.lang == "tr" else 1
lang_choice = st.sidebar.radio(t["lang_select_label"], ["Turkce", "English"], index=lang_index)
if (lang_choice == "Turkce" and st.session_state.lang != "tr") or \
   (lang_choice == "English" and st.session_state.lang != "en"):
    st.session_state.lang = "tr" if lang_choice == "Turkce" else "en"
    st.rerun() 
dark = st.sidebar.toggle(t["dark_mode_label"], value=st.session_state.dark)
st.session_state.dark = dark
st.sidebar.divider()

# =========================================
# THEMES + GRADIENT BACKGROUND
# =========================================
if st.session_state.dark:
    bg_gradient = "background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);"
    text_color = "#E0F7FA"
    card = "#132E35"
    accent = "#00B8BE"
else:
    bg_gradient = "background: linear-gradient(135deg, #1A3C40, #165E63, #1A3C40);"
    text_color = "#E6FAFA"
    card = "#244D52"
    accent = "#00C2CB"

st.markdown(f"""
<style>
@keyframes gradientMove {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
html, body, [class*="stApp"], section.main {{
    {bg_gradient}
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    color: {text_color} !important;
    font-family: 'Inter', sans-serif;
}}
.stButton>button {{
    background: {accent} !important; color: white !important; border-radius: 8px !important; border: none !important;
    padding: 10px 22px !important; font-weight: 600 !important;
}}
.stButton>button:hover {{ background: #00E0E7 !important; transform: scale(1.03); }}
.stDataEditor, .stDataFrame {{
    background: {card} !important; border-radius: 10px; border: 1px solid {accent}; color: {text_color} !important;
}}
div[data-testid="stTabs"] > div > div > button {{
    background-color: {card} !important; color: {text_color} !important; border-radius: 12px !important;
    border: 2px solid {accent} !important; font-size: 18px !important; padding: 10px 20px !important;
    margin-right: 8px !important; font-weight: 600 !important;
}}
.analysis-box {{
    background-color: {card}; border: 1px solid {accent}; border-radius: 10px;
    padding: 16px; margin-top: 20px;
}}
</style>
""", unsafe_allow_html=True)

# =========================================
# NAVIGATION TABS (ESKI YAPIYA DONDUK)
# =========================================
tabs = st.tabs(t["tabs"])

# =========================================
# HOME TAB (GUNCEL REHBER EKLENDI)
# =========================================
with tabs[0]:
    st.markdown(f"""
    <div style='text-align:center;'>
        <img src="https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png" width="220">
        <h2>{t["home_title"]}</h2>
        <p style='font-size:18px;'>{t["home_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader(t["home_guide_header"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {t['home_guide_1_header']}")
        st.markdown(t['home_guide_1_text'])
    with col2:
        st.markdown(f"### {t['home_guide_2_header']}")
        st.markdown(t['home_guide_2_text'])
    with col3:
        st.markdown(f"### {t['home_guide_3_header']}")
        st.markdown(t['home_guide_3_text'])
    
    st.divider()
    st.markdown(f"<h4 style='text-align: center;'>{t['home_guide_4_header']}</h4>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{t['home_guide_4_text']}</p>", unsafe_allow_html=True)


# =========================================
# DATA TAB (ORNEK VERI BUTONU EKLENDI)
# =========================================
with tabs[1]:
    st.subheader(t["data_header"])
    
    uploaded_file = st.file_uploader(t["data_upload_label"], type=["csv", "xlsx"], help=t["data_upload_help"])
    
    st.divider()
    if st.button(t["data_sample_button"]):
        with st.spinner(t["data_sample_loading"]):
            time.sleep(1) # Gercekmis gibi hissettir
            st.session_state.user_data = create_sample_data()
            st.session_state.model = None # Yeni veri yuklendiginde eski modeli sil
            st.success(t["data_upload_success"].format(rows=len(st.session_state.user_data)))
            st.rerun() # Veri onizlemesini guncellemek icin

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            if all(col in df.columns for col in REQUIRED_COLS):
                st.session_state.user_data = df
                st.session_state.model = None # Yeni veri yuklendiginde eski modeli sil
                st.success(t["data_upload_success"].format(rows=len(df)))
            else:
                st.error(t["data_upload_help"])
                st.session_state.user_data = None
                
        except Exception as e:
            st.error(f"{t['data_upload_error']} ({e})")
            st.session_state.user_data = None
    
    # Veri Onizlemesi (Eger veri yuklenmisse goster)
    if st.session_state.user_data is not None:
        st.subheader(t["data_preview_header"])
        st.dataframe(st.session_state.user_data[REQUIRED_COLS].head(10))
    elif uploaded_file is None:
        st.info(t["data_upload_info"])


# =========================================
# MODEL TAB (AKILLI MODELE GUNCELENDI)
# =========================================
with tabs[2]:
    st.subheader(t["model_header"])
    
    if st.session_state.user_data is None:
        st.warning(t["model_warning_upload"])
    else:
        st.success(f"Eğitim için {len(st.session_state.user_data)} satır veri yüklendi.")
        
        if st.button(t["model_button"]):
            with st.spinner(t["model_spinner"]):
                time.sleep(1)
                
                X = st.session_state.user_data[FEATURES]
                y = st.session_state.user_data[TARGET]
                
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                except ValueError:
                    X_train, X_test, y_train, y_test = X, pd.DataFrame(), y, pd.Series()

                model = RandomForestRegressor(n_estimators=150, random_state=42)
                model.fit(X_train, y_train)
                
                if not X_test.empty:
                    preds = model.predict(X_test)
                    r2 = round(r2_score(y_test, preds), 3)
                    mae = round(mean_absolute_error(y_test, preds), 3)
                    st.success(t["model_success_metrics"].format(r2=r2, mae=mae))
                else:
                    st.success(t["model_success_no_test"])

            os.makedirs("models", exist_ok=True)
            joblib.dump(model, "models/karmentor_model.joblib")
            st.session_state.model = model
            st.balloons()

# =========================================
# PREDICTION TAB (AKILLI MODELE GUNCELENDI)
# =========================================
with tabs[3]:
    st.subheader(t["predict_header"])

    if not st.session_state.model:
        st.warning(t["predict_warning_model"])
    elif not AI_AKTIF:
         st.warning(t["predict_warning_api"])
    else:
        
        # === 1. Bolum: Tekli Tahmin ===
        col1, col2 = st.columns(2)
        with col1:
            fiyat = st.number_input(t["predict_input_price"], value=120.0, step=10.0)
            maliyet = st.number_input(t["predict_input_cost"], value=70.0, step=5.0)
        with col2:
            indirim = st.slider(t["predict_input_discount"], 0.0, 0.5, 0.1, format="%.2f")

        if st.button(t["predict_button_calculate"]):
            yeni = pd.DataFrame([[fiyat, maliyet, indirim]], columns=FEATURES) 
            
            model = st.session_state.model
            tahmini_satis = model.predict(yeni)[0]
            tahmini_satis = max(0, int(tahmini_satis)) 
            
            kar_tutar = (fiyat - maliyet) * tahmini_satis * (1 - indirim)
            
            st.session_state.son_tahmin = {
                "fiyat": fiyat, "maliyet": maliyet, "satis": tahmini_satis,
                "indirim": indirim, "tutar": kar_tutar
            }

            st.metric(t["predict_metric_profit_rate"], f"{tahmini_satis} adet")
            st.metric(t["predict_metric_profit_amount"], f"{kar_tutar:,.2f} TL")

            fiyat_araligi = np.linspace(fiyat * 0.7, fiyat * 1.3, 20)
            if len(fiyat_araligi) < 2: 
                fiyat_araligi = [fiyat * 0.9, fiyat, fiyat * 1.1] 
                
            tahmini_satis_listesi = []
            for f in fiyat_araligi:
                yeni_grafik = pd.DataFrame([[f, maliyet, indirim]], columns=FEATURES) 
                tahmini_satis_listesi.append(max(0, model.predict(yeni_grafik)[0]))

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(fiyat_araligi, tahmini_satis_listesi, color=accent, linewidth=3, marker="o")
            ax.set_title(t["predict_plot_title"], color=text_color)
            ax.set_xlabel(t["predict_plot_xlabel"], color=text_color)
            ax.set_ylabel(t["predict_plot_ylabel"], color=text_color)
            ax.tick_params(colors=text_color)
            fig.patch.set_facecolor("none") 
            ax.set_facecolor(card)
            st.pyplot(fig)

        st.divider()

        # ===================================
        # 2. Bolum: OPTIMIZASYON (AKILLI)
        # ===================================
        st.subheader(t["optim_header"])
        st.caption(t["optim_help"])
        
        # Girdileri ustteki bolumden alir (fiyat, maliyet, indirim)
        # Eger 'Kari Hesapla'ya basilmadiysa, varsayilanlari kullan
        maliyet_opt = maliyet if 'maliyet' in locals() and 'fiyat' in locals() else 70.0
        indirim_opt = indirim if 'indirim' in locals() and 'fiyat' in locals() else 0.1
        fiyat_varsayilan = fiyat if 'fiyat' in locals() else 120.0
        
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            min_fiyat = st.number_input(t["optim_min_price"], value=fiyat_varsayilan*0.7, step=5.0)
        with opt_col2:
            max_fiyat = st.number_input(t["optim_max_price"], value=fiyat_varsayilan*1.3, step=5.0)
        
        if st.button(t["optim_button"]):
            if max_fiyat <= min_fiyat:
                st.error("Maksimum fiyat, minimum fiyattan buyuk olmalidir.")
            else:
                with st.spinner(t["optim_spinner"]):
                    model = st.session_state.model
                    
                    adim = max(1, int((max_fiyat - min_fiyat) / 100))
                    fiyat_listesi = range(int(min_fiyat), int(max_fiyat) + 1, adim)
                    
                    best_profit = -float('inf')
                    best_price = min_fiyat
                    best_sales = 0
                    
                    opt_results = []
                    
                    for p in fiyat_listesi:
                        test_df = pd.DataFrame([[p, maliyet_opt, indirim_opt]], columns=FEATURES)
                        tahmini_satis = model.predict(test_df)[0]
                        tahmini_satis = max(0, tahmini_satis) 
                        
                        current_profit_amount = (p - maliyet_opt) * tahmini_satis * (1 - indirim_opt)
                        
                        opt_results.append((p, current_profit_amount))
                        
                        if current_profit_amount > best_profit:
                            best_profit = current_profit_amount
                            best_price = p
                            best_sales = tahmini_satis
                            
                    st.success(t["optim_success_title"])
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric(t["optim_success_metric_price"], f"{best_price:,.2f} TL")
                    res_col2.metric(t["optim_success_metric_profit"], f"{best_profit:,.2f} TL")
                    res_col3.metric(t["optim_success_metric_sales"], f"{int(best_sales)} adet")
                    
                    opt_df = pd.DataFrame(opt_results, columns=["Fiyat", "Tahmini Kar"])
                    st.line_chart(opt_df.set_index("Fiyat"), color=accent)

        st.divider()

        # ===================================
        # 3. Bolum: AI ANALIZI (GUNCEL PROMPT)
        # ===================================
        if "son_tahmin" in st.session_state and AI_AKTIF:
            st.subheader(t["predict_analysis_header"])
            
            if st.button(t["predict_analysis_button"]):
                with st.spinner(t["predict_analysis_spinner"]):
                    try:
                        tahmin_verisi = st.session_state.son_tahmin
                        
                        prompt = t["predict_analysis_prompt_user"].format(
                            fiyat=tahmin_verisi['fiyat'],
                            maliyet=tahmin_verisi['maliyet'],
                            indirim=f"{tahmin_verisi['indirim']*100:.0f}",
                            satis=tahmin_verisi['satis'], # Bu artik tahmini satis adedi
                            tutar=f"{tahmin_verisi['tutar']:,.2f}"
                        )
                        
                        response = client.chat.completions.create(
                            extra_headers={"HTTP-Referer": "https.karmentor.streamlit.app", "X-Title": "KarMentor"},
                            model="openai/gpt-oss-20b:free", 
                            messages=[
                                {"role": "system", "content": t["predict_analysis_prompt_system"]},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        analysis_text = response.choices[0].message.content
                        
                        st.markdown(f"""<div class="analysis-box"><p style='color:{text_color};'>{analysis_text}</p></div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"{t['predict_analysis_error']} {e}")
        
# =========================================
# CHATBOT (HARUNCUK)
# =========================================
col1, col2, col3 = st.columns([10, 10, 3]) 
with col3:
    with st.popover(t["chatbot_popover_label"]):
        
        if st.button(t["chatbot_clear_button"]):
            st.session_state.haruncuk_messages = []
            st.rerun() 

        for message in st.session_state.haruncuk_messages:
            avatar_img = HARUNCUK_AVATAR_URL if message["role"] == "assistant" else "🧑‍💻"
            with st.chat_message(message["role"], avatar=avatar_img): 
                st.markdown(message["content"])

        if prompt := st.chat_input(t["chatbot_input_placeholder"]):
            
            st.session_state.haruncuk_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            system_prompt_content = t["chatbot_system_prompt"]
            system_prompt = {"role": "system", "content": system_prompt_content}
            messages_for_api = [system_prompt] + st.session_state.haruncuk_messages

            with st.chat_message("assistant", avatar=HARUNCUK_AVATAR_URL):
                try:
                    with st.spinner(t["chatbot_thinking"]):
                        stream = client.chat.completions.create(
                            extra_headers={"HTTP-Referer": "https.karmentor.streamlit.app", "X-Title": "KarMentor"},
                            model="openai/gpt-oss-20b:free",
                            messages=messages_for_api,
                            stream=True
                        )
                        response_content = st.write_stream(stream)
                    st.session_state.haruncuk_messages.append({"role": "assistant", "content": response_content})
                except Exception as e:
                    st.error(f"{t['chatbot_error']} {e}")

# =========================================
# FOOTER
# =========================================
st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    f"<p style='text-align:center;'>{t['footer_text']}</p>",
    unsafe_allow_html=True
)