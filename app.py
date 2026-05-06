import streamlit as st
import pandas as pd
import joblib
import numpy as np
import neattext.functions as nfx
from textblob import TextBlob
from tensorflow.keras.models import load_model

# 1. Gerekli Fonksiyonların Tanımlanması 
def clean_text(text):
    text = text.lower() 
    text = nfx.remove_special_characters(text) 
    text = nfx.remove_multiple_spaces(text) 
    return text

def ekkok(text):
    words = TextBlob(text).words #
    return [word.lemmatize() for word in words] 

# 2. Model ve Vektörleştiricinin Yüklenmesi
@st.cache_resource
def load_assets():
    model = load_model('llm_detector_model.h5') #
    vect = joblib.load('vectorizer.pkl') 
    return model, vect

# Sayfa Yapılandırması
st.set_page_config(page_title="AI vs Human Text Detector", page_icon="🤖")

st.title("🤖 AI vs Human Text Detector")
st.markdown("""
Bu uygulama, girilen metnin bir **İnsan** tarafından mı yoksa **Yapay Zeka (LLM)** tarafından mı yazıldığını tahmin eder.
""")

try:
    model, vect = load_assets()
    
    # Kullanıcı Girişi
    user_input = st.text_area("Analiz edilecek metni buraya yapıştırın:", height=250)

    if st.button("Tahmin Et"):
        if user_input:
            # 3. Ön İşleme ve Tahmin Adımları
            cleaned_text = clean_text(user_input)
            # Not: Sadece transform kullanıyoruz
            input_vect = vect.transform([cleaned_text]).toarray() 
            
            prediction_prob = model.predict(input_vect)[0][0] 
            
            # Sonuçların Gösterilmesi
            st.subheader("Analiz Sonucu")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction_prob > 0.5:
                    st.error(f"Tahmin: YAPAY ZEKA (AI)")
                else:
                    st.success(f"Tahmin: İNSAN (Human)")
            
            with col2:
                st.info(f"AI Olma Olasılığı: %{prediction_prob * 100:.2f}")
            
            # Detaylı grafik
            st.progress(float(prediction_prob))
            
        else:
            st.warning("Lütfen bir metin girin.")

except Exception as e:
    st.error(f"Dosyalar yüklenirken hata oluştu: {e}")
    st.info("Lütfen 'llm_detector_model.keras' ve 'vectorizer.pkl' dosyalarının uygulama dizininde olduğundan emin olun.")