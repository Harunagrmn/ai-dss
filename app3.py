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

# OpenAI kütüphanesini kullaniyoruz (OpenRouter icin)
from openai import OpenAI

# =========================================
# DIL DESTEGI (EMOJILER EKLENDI)
# =========================================
translations = {
    "tr": {
        "page_title": "KarMentor | AI Profit Assistant",
        "sidebar_title": "⚙️ KarMentor Menu",
        "lang_select_label": "🌍 Language / Dil",
        "lang_options": ["Turkce", "English"],
        "dark_mode_label": "🌗 Dark Mode",
        "chatbot_popover_label": "🤖 Haruncuk'a Soru Sor",
        "chatbot_input_placeholder": "Bana bir soru sor...",
        "chatbot_thinking": "Haruncuk dusunuyor...",
        "chatbot_error": "Haruncuk bir hatayla karsilasti:",
        "chatbot_clear_button": "Sohbeti Temizle",
        "chatbot_system_prompt": """Sen Haruncuk adinda, KarMentor adli bir kar tahmin uygulamasinda kullanicilara yardimci olan bir asistansin.
Gorevin, kullanicilarin uygulamanin ozelliklerini anlamasina yardimci olmaktir.
Ornegin: 'Birim Fiyat nedir?', 'Model egitimi nasil calisir?', 'Tahmin sekmesi ne ise yarar?' gibi sorulara cevap ver.
Cevaplarin yardimci, net ve kisa olsun. Sadece Turkce konus.""",
        "tabs": ["🏠 Ana Sayfa", "📋 Veri", "🧠 Model", "💰 Tahmin"],
        "home_title": "💼 KarMentor - AI Profit Decision Assistant",
        "home_subtitle": "Verilerinizi analiz edin, modelinizi egitin ve karliliginizi tahmin edin 🚀",
        "data_header": "📋 Veri Duzenleme",
        "data_caption": "Verileri duzenleyebilir veya yenilerini ekleyebilirsin.",
        "model_header": "🧠 Model Egitimi",
        "model_warning_rows": "Modeli egitmek icin en az 3 satir veri gereklidir.",
        "model_button": "🧠 Modeli Egit ve Aktiflestir",
        "model_spinner": "🔄 Model egitiliyor...",
        "model_success_metrics": "✅ R²: {r2} | MAE: {mae}",
        "model_success_no_test": "✅ Model egitildi (Test verisi yetersiz, tum veri kullanildi).",
        "predict_header": "💰 Kar Tahmini ve Gorsellestirme",
        "predict_warning_model": "Lutfen once 'Model' sekmesinden modeli egitin.",
        "predict_warning_api": "Yapay Zeka (OpenRouter) API anahtari baslatilamadi. Lutfen ustteki hata mesajini ve secrets dosyanizi kontrol edin.",
        "predict_input_price": "Birim Fiyat (TL)",
        "predict_input_cost": "Birim Maliyet (TL)",
        "predict_input_quantity": "Satis Adedi",
        "predict_input_discount": "Indirim Orani",
        "predict_button_calculate": "📈 Kari Hesapla",
        "predict_metric_profit_rate": "💸 Tahmini Kar Orani (%)",
        "predict_metric_profit_amount": "💰 Tahmini Kar Tutari (TL)",
        "predict_plot_title": "💹 Fiyat - Kar Orani Grafigi",
        "predict_plot_xlabel": "Birim Fiyat (TL)",
        "predict_plot_ylabel": "Kar Orani (%)",
        "predict_analysis_header": "🤖 OpenRouter Yapay Zeka Analizi",
        "predict_analysis_button": "💡 Bu Senaryo Icin Analiz Al",
        "predict_analysis_spinner": "🔄 Ucretsiz model bu senaryoyu analiz ediyor...",
        "predict_analysis_error": "OpenRouter ile analiz yapilirken bir hata olustu:",
        "predict_analysis_prompt_system": "Sen bir finansal analiz asistanisin.",
        "predict_analysis_prompt_user": """
Sen KarMentor adli bir is zekasi uygulamasinin finansal analiz asistanisin.
Rolun, bir makine ogrenimi modelinin (RandomForest) tahminini alip,
bunu CEO'larin anlayacagi dilde, eyleme gecirilebilir bir is tavsiyesine donusturmektir.
Asagidaki senaryoyu analiz et:
Girdiler:
- Birim Fiyat (TL): {fiyat}
- Birim Maliyet (TL): {maliyet}
- Satis Adedi: {satis}
- Indirim Orani: {indirim}%
Model Tahmini:
- Tahmini Kar Orani: {oran}%
- Tahmini Toplam Kar (TL): {tutar}
Lutfen bu senaryoya dayanarak kisa (3-4 cumlelik) bir analiz ve stratejik bir oneri sun.
Cevabin sadece analizin kendisi olsun, "Elbet ki, iste analiz:" gibi baslama.
""",
        "footer_text": "© 2025 KarMentor | Gelistiren: Harun Agirman"
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
        "chatbot_system_prompt": """You are Haruncuk, a helpful assistant in the KarMentor profit prediction app.
Your job is to help users understand the app's features.
For example: Answer questions like 'What is Unit Price?', 'How does model training work?', 'What does the Prediction tab do?'.
Keep your answers helpful, clear, and brief. Speak only in English.""",
        "tabs": ["🏠 Home", "📋 Data", "🧠 Model", "💰 Prediction"],
        "home_title": "💼 KarMentor - AI Profit Decision Assistant",
        "home_subtitle": "Analyze your data, train your model, and predict your profitability 🚀",
        "data_header": "📋 Data Editor",
        "data_caption": "You can edit the data or add new rows.",
        "model_header": "🧠 Model Training",
        "model_warning_rows": "At least 3 rows of data are required to train the model.",
        "model_button": "🧠 Train and Activate Model",
        "model_spinner": "🔄 Model is training...",
        "model_success_metrics": "✅ R²: {r2} | MAE: {mae}",
        "model_success_no_test": "✅ Model trained (Insufficient test data, all data was used).",
        "predict_header": "💰 Profit Prediction and Visualization",
        "predict_warning_model": "Please train the model on the 'Model' tab first.",
        "predict_warning_api": "AI (OpenRouter) API key could not be initialized. Please check the error message above and your secrets file.",
        "predict_input_price": "Unit Price (TL)",
        "predict_input_cost": "Unit Cost (TL)",
        "predict_input_quantity": "Sales Quantity",
        "predict_input_discount": "Discount Rate",
        "predict_button_calculate": "📈 Calculate Profit",
        "predict_metric_profit_rate": "💸 Estimated Profit Rate (%)",
        "predict_metric_profit_amount": "💰 Estimated Profit Amount (TL)",
        "predict_plot_title": "💹 Price - Profit Rate Graph",
        "predict_plot_xlabel": "Unit Price (TL)",
        "predict_plot_ylabel": "Profit Rate (%)",
        "predict_analysis_header": "🤖 OpenRouter AI Analysis",
        "predict_analysis_button": "💡 Get Analysis for This Scenario",
        "predict_analysis_spinner": "🔄 Free model is analyzing this scenario...",
        "predict_analysis_error": "An error occurred while analyzing with OpenRouter:",
        "predict_analysis_prompt_system": "You are a financial analysis assistant.",
        "predict_analysis_prompt_user": """
You are a financial analysis assistant for an app called KarMentor.
Your role is to take the output from a machine learning model (RandomForest) and
translate it into an actionable business recommendation in plain language for a CEO.
Analyze the following scenario:
Inputs:
- Unit Price (TL): {fiyat}
- Unit Cost (TL): {maliyet}
- Sales Quantity: {satis}
- Discount Rate: {indirim}%
Model Prediction:
- Estimated Profit Rate: {oran}%
- Estimated Total Profit (TL): {tutar}
Please provide a brief (3-4 sentences) analysis and one strategic recommendation based on this scenario.
Your response should be only the analysis itself, do not start with "Certainly, here is the analysis:".
""",
        "footer_text": "© 2025 KarMentor | Developed by: Harun Agirman"
    }
}


# =========================================
# PAGE SETTINGS
# =========================================
# Dil secimini session state'e gore yap
if "lang" not in st.session_state:
    st.session_state.lang = "tr" # Varsayilan dil Turkce

# Secilen dile gore metinleri yukle
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
if "veri" not in st.session_state:
    st.session_state.veri = pd.DataFrame([
        [100, 60, 20, 0.10, 0.40],
        [120, 70, 25, 0.05, 0.42],
        [150, 80, 30, 0.00, 0.50],
        [90,  55, 18, 0.15, 0.33],
        [130, 75, 28, 0.07, 0.46],
    ], columns=["Birim Fiyat", "Birim Maliyet", "Satis Adedi", "Indirim Orani", "Kar Orani"])

if "haruncuk_messages" not in st.session_state:
    st.session_state.haruncuk_messages = []


# =========================================
# SIDEBAR (ICON + ANIMATION)
# =========================================
st.sidebar.markdown("""
<style>
/* ... (CSS stillari ayni kaliyor) ... */
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

# !!! DUZELTME BURADA !!!
# unsafe_allow_html=True parametresini ekledik.
st.sidebar.markdown(f"<div class='sidebar-title'>{t['sidebar_title']}</div>", unsafe_allow_html=True)

# DIL SECENEGI (IYILESTIRILDI)
lang_index = 0 if st.session_state.lang == "tr" else 1
lang_choice = st.sidebar.radio(t["lang_select_label"], ["Turkce", "English"], index=lang_index)

# Dil degistiginde sayfayi yeniden yukle
if (lang_choice == "Turkce" and st.session_state.lang != "tr") or \
   (lang_choice == "English" and st.session_state.lang != "en"):
    st.session_state.lang = "tr" if lang_choice == "Turkce" else "en"
    # Sayfayi yeniden yuklemesi icin st.rerun() kullaniyoruz
    st.rerun() 

# Dark mode
dark = st.sidebar.toggle(t["dark_mode_label"], value=st.session_state.dark)
st.session_state.dark = dark

st.sidebar.divider()

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
/* ... (CSS stillari ayni kaliyor, analysis-box'i da iceriyor) ... */
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
.analysis-box {{
    background-color: {card};
    border: 1px solid {accent};
    border-radius: 10px;
    padding: 16px;
    margin-top: 20px;
}}
</style>
""", unsafe_allow_html=True)

# =========================================
# NAVIGATION TABS
# =========================================
tabs = st.tabs(t["tabs"])

# =========================================
# HOME TAB
# =========================================
with tabs[0]:
    st.markdown(f"""
    <div style='text-align:center;'>
        <img src="https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/karmentor_logo1.png" width="220">
        <h2>{t["home_title"]}</h2>
        <p style='font-size:18px;'>{t["home_subtitle"]}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# DATA TAB (IYILESTIRILDI)
# =========================================
with tabs[1]:
    st.subheader(t["data_header"])
    
    current_lang_is_tr = st.session_state.lang == "tr"
    
    if current_lang_is_tr:
        kolon_adlari = ["Birim Fiyat", "Birim Maliyet", "Satis Adedi", "Indirim Orani", "Kar Orani"]
    else:
        kolon_adlari = ["Unit Price", "Unit Cost", "Sales Quantity", "Discount Rate", "Profit Rate"]
    
    original_kolon_adlari = ["Birim Fiyat", "Birim Maliyet", "Satis Adedi", "Indirim Orani", "Kar Orani"]

    display_veri = st.session_state.veri.copy()
    display_veri.columns = kolon_adlari
    
    edited_data = st.data_editor(display_veri, num_rows="dynamic", use_container_width=True, key=f"data_editor_{st.session_state.lang}")
    st.caption(t["data_caption"])
    
    try:
        edited_data_original_cols = edited_data.copy()
        edited_data_original_cols.columns = original_kolon_adlari
        
        if not st.session_state.veri.equals(edited_data_original_cols):
            st.session_state.veri = edited_data_original_cols
            st.rerun() 
    except Exception as e:
        pass


# =========================================
# MODEL TAB
# =========================================
with tabs[2]:
    st.subheader(t["model_header"])
    if len(st.session_state.veri) < 3:
        st.warning(t["model_warning_rows"])
    else:
        if st.button(t["model_button"]):
            with st.spinner(t["model_spinner"]):
                time.sleep(1)
                X = st.session_state.veri.drop(columns=["Kar Orani"])
                y = st.session_state.veri["Kar Orani"]
                
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

# =========================================
# PREDICTION TAB
# =========================================
with tabs[3]:
    st.subheader(t["predict_header"])

    if not st.session_state.model:
        st.warning(t["predict_warning_model"])
    elif not AI_AKTIF:
         st.warning(t["predict_warning_api"])
    else:
        col1, col2 = st.columns(2)
        with col1:
            fiyat = st.number_input(t["predict_input_price"], value=120.0, step=10.0)
            maliyet = st.number_input(t["predict_input_cost"], value=70.0, step=5.0)
        with col2:
            satis = st.number_input(t["predict_input_quantity"], value=25, step=1)
            indirim = st.slider(t["predict_input_discount"], 0.0, 0.5, 0.1, format="%.2f")

        if st.button(t["predict_button_calculate"]):
            yeni = pd.DataFrame([[fiyat, maliyet, satis, indirim]],
                                columns=original_kolon_adlari[:4]) # Orijinal adlari kullan
            
            model = st.session_state.model
            oran = model.predict(yeni)[0]
            kar_tutar = (fiyat - maliyet) * satis * (1 - indirim)
            
            st.session_state.son_tahmin = {
                "fiyat": fiyat,
                "maliyet": maliyet,
                "satis": satis,
                "indirim": indirim,
                "oran": oran,
                "tutar": kar_tutar
            }

            st.metric(t["predict_metric_profit_rate"], f"{oran*100:.2f}")
            st.metric(t["predict_metric_profit_amount"], f"{kar_tutar:,.2f}")

            fiyat_araligi = list(range(int(fiyat*0.7), int(fiyat*1.3), 5))
            if not fiyat_araligi: 
                fiyat_araligi = [fiyat]
                
            tahminler = []
            for f in fiyat_araligi:
                yeni_grafik = pd.DataFrame([[f, maliyet, satis, indirim]],
                                    columns=original_kolon_adlari[:4]) # Orijinal adlari kullan
                tahminler.append(model.predict(yeni_grafik)[0]*100)

            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(fiyat_araligi, tahminler, color=accent, linewidth=3, marker="o")
            ax.set_title(t["predict_plot_title"], color=text_color)
            ax.set_xlabel(t["predict_plot_xlabel"], color=text_color)
            ax.set_ylabel(t["predict_plot_ylabel"], color=text_color)
            ax.tick_params(colors=text_color)
            fig.patch.set_facecolor("none") 
            ax.set_facecolor(card)
            st.pyplot(fig)


        st.divider()

        # ===================================
        # OPENROUTER (GPT) ANALIZ BOLUMU
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
                            satis=tahmin_verisi['satis'],
                            indirim=f"{tahmin_verisi['indirim']*100:.0f}",
                            oran=f"{tahmin_verisi['oran']*100:.2f}",
                            tutar=f"{tahmin_verisi['tutar']:,.2f}"
                        )
                        
                        response = client.chat.completions.create(
                            extra_headers={
                                "HTTP-Referer": "https://karmentor.streamlit.app",
                                "X-Title": "KarMentor"
                            },
                            model="openai/gpt-oss-20b:free", 
                            messages=[
                                {"role": "system", "content": t["predict_analysis_prompt_system"]},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        
                        analysis_text = response.choices[0].message.content
                        
                        st.markdown(f"""
                        <div class="analysis-box">
                            <p style='color:{text_color};'>{analysis_text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"{t['predict_analysis_error']} {e}")
        

# =========================================
# YENI CHATBOT KONUMU (SAG ALT POPOVER)
# =========================================

# !!! DUZELTME BURADA: Haruncuk'un avatar URL'si !!!
# LUTFEN .png uzantisini, eger dosyan .jpg veya baska bir seyse, onunla degistir.
HARUNCUK_AVATAR_URL = "https://raw.githubusercontent.com/Harunagrmn/ai-dss/main/assets/haruncukbot.png"

# Konumu duzenlemek icin 2 bos sutun ve 1 popover sutunu
col1, col2, col3 = st.columns([10, 10, 3]) # Bosluk
with col3:
    with st.popover(t["chatbot_popover_label"]):
        
        # Sohbeti Temizle butonu
        if st.button(t["chatbot_clear_button"]):
            st.session_state.haruncuk_messages = []
            st.rerun() 

        # 1. Sohbet gecmisini goster
        for message in st.session_state.haruncuk_messages:
            # !!! DUZELTME BURADA: Avatarlar eklendi !!!
            avatar_img = HARUNCUK_AVATAR_URL if message["role"] == "assistant" else "🧑‍💻"
            with st.chat_message(message["role"], avatar=avatar_img): 
                st.markdown(message["content"])

        # 2. Kullanicidan yeni girdi al
        if prompt := st.chat_input(t["chatbot_input_placeholder"]):
            
            # 3. Kullanicinin mesajini gecmise ve ekrana ekle
            st.session_state.haruncuk_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            # 4. Yapay zekaya gondermek icin mesaj listesini hazirla
            system_prompt_content = t["chatbot_system_prompt"]
            system_prompt = {"role": "system", "content": system_prompt_content}
            messages_for_api = [system_prompt] + st.session_state.haruncuk_messages

            # 5. OpenRouter'a baglan ve cevabi stream et
            with st.chat_message("assistant", avatar=HARUNCUK_AVATAR_URL): # !!! DUZELTME BURADA !!!
                try:
                    with st.spinner(t["chatbot_thinking"]):
                        stream = client.chat.completions.create(
                            extra_headers={
                                "HTTP-Referer": "https://karmentor.streamlit.app",
                                "X-Title": "KarMentor"
                            },
                            model="openai/gpt-oss-20b:free",
                            messages=messages_for_api,
                            stream=True
                        )
                        response_content = st.write_stream(stream)
                    
                    # 6. Tam cevabi sohbet gecmisine ekle
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