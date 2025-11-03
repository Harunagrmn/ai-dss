# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import time
from openai import OpenAI

# =========================================
# DIL DESTEGI (TÜRKÇE KARAKTERLER DÜZELTİLDİ)
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
        "chatbot_system_prompt": "Sen Haruncuk adında, KarMentor adlı bir kâr tahmin uygulamasında kullanıcılara yardımcı olan bir asistansın...", # (Kısa tuttum)
        "home_title": "💼 KarMentor - Yapay Zeka Kâr Destek Sistemi",
        "home_subtitle": "Verilerinizi analiz edin, modelinizi eğitin ve kârlılığınızı optimize edin 🚀",
        
        # --- YENI EKLENDI: ANA SAYFA REHBERI ---
        "home_guide_header": "Uygulama Nasıl Kullanılır? (Hızlı Rehber)",
        "home_guide_1_header": "Adım 1: 📋 Veri Yükleme",
        "home_guide_1_text": "Kenar çubuğundaki '1_Veri_Yukleme' sayfasına gidin. 'Birim Fiyat', 'Birim Maliyet', 'Indirim Orani' ve 'Satis Adedi' kolonlarını içeren kendi .csv veya .xlsx dosyanızı yükleyin.",
        "home_guide_2_header": "Adım 2: 🧠 Model Eğitimi",
        "home_guide_2_text": "Kenar çubuğundaki '2_Model_Egitimi' sayfasına gidin. Yüklediğiniz veriyi kullanarak 'Satis Adedi'ni tahmin edecek yapay zeka modelinizi eğitin.",
        "home_guide_3_header": "Adım 3: 💰 Tahmin ve Optimizasyon",
        "home_guide_3_text": "Kenar çubuğundaki '3_Tahmin_ve_Optimizasyon' sayfasına gidin. Eğittiğiniz modeli kullanarak tekli tahminler yapın veya 'En İyi Fiyatı Bul' optimizasyon aracını çalıştırın.",
        "home_guide_4_header": "Yardıma mı İhtiyacınız Var? 🤖",
        "home_guide_4_text": "Herhangi bir adımda takılırsanız veya 'Birim Fiyat nedir?' gibi bir sorunuz olursa, sağ alttaki 'Haruncuk'a Soru Sor' butonunu kullanmaktan çekinmeyin!",

        "footer_text": "© 2025 KarMentor | Geliştiren: Harun Ağırman"
        # (Diger sayfalara ait metinleri bu dosyadan sildim, gereksiz)
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
        "chatbot_system_prompt": "You are Haruncuk, a helpful assistant...",
        "home_title": "💼 KarMentor - AI Profit Decision Assistant",
        "home_subtitle": "Analyze your data, train your model, and optimize your profitability 🚀",

        # --- NEW: HOMEPAGE GUIDE ---
        "home_guide_header": "How to Use This App (Quick Guide)",
        "home_guide_1_header": "Step 1: 📋 Data Upload",
        "home_guide_1_text": "Go to the '1_Data_Upload' page in the sidebar. Upload your own .csv or .xlsx file containing 'Birim Fiyat', 'Birim Maliyet', 'Indirim Orani', and 'Satis Adedi' columns.",
        "home_guide_2_header": "Step 2: 🧠 Model Training",
        "home_guide_2_text": "Go to the '2_Model_Training' page. Use the data you uploaded to train your AI model to predict 'Sales Quantity'.",
        "home_guide_3_header": "Step 3: 💰 Prediction & Optimization",
        "home_guide_3_text": "Go to the '3_Prediction_and_Optimization' page. Make single predictions or run the 'Find Best Price' optimization tool using your trained model.",
        "home_guide_4_header": "Need Help? 🤖",
        "home_guide_4_text": "If you get stuck at any step, or have a question like 'What is Unit Price?', don't hesitate to use the 'Ask Haruncuk' button in the bottom-right corner!",
        
        "footer_text": "© 2025 KarMentor | Developed by: Harun Agirman"
    }
}
HARUNCUK_AVATAR_URL = "https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/haruncukbot.png"

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
    st.error(f"**Yapay Zeka (OpenRouter) Baslatilamadi!** Hata: {e}")

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
/* ... (Tum CSS kodlari buraya) ... */
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
# ANA SAYFA ICERIGI (YENI REHBER EKLENDI)
# =========================================
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
    st.markdown(f"**{t['home_guide_1_header']}**")
    st.markdown(t['home_guide_1_text'])
with col2:
    st.markdown(f"**{t['home_guide_2_header']}**")
    st.markdown(t['home_guide_2_text'])
with col3:
    st.markdown(f"**{t['home_guide_3_header']}**")
    st.markdown(t['home_guide_3_text'])

st.divider()
st.markdown(f"<h4 style='text-align: center;'>{t['home_guide_4_header']}</h4>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{t['home_guide_4_text']}</p>", unsafe_allow_html=True)


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