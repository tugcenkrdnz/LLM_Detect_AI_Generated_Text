# 🤖 LLM Detect: AI vs. Human Text Classification

Bu proje, bir metnin bir **insan** tarafından mı yazıldığını yoksa bir **Büyük Dil Modeli (LLM)** tarafından mı üretildiğini tespit eden uçtan uca bir Doğal Dil İşleme (NLP) ve Derin Öğrenme çözümüdür. Proje, Kaggle üzerindeki [LLM Detect AI-Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text) yarışması temel alınarak geliştirilmiştir.

## 📋 Proje Özeti
Yapay zeka modellerinin (ChatGPT, Claude vb.) yaygınlaşmasıyla birlikte, akademik dürüstlüğü korumak ve içerik özgünlüğünü doğrulamak kritik bir ihtiyaç haline gelmiştir. Bu çalışma, karmaşık metin yapılarını analiz ederek yapay zekanın "robotik" dil kalıpları ile insanların "esnek ve hatalı" yazım tarzlarını ayırt eder.

## 🛠️ Kullanılan Teknolojiler
*   **Programlama Dili:** Python
*   **Veri Analizi:** Pandas, Numpy
*   **NLP Araçları:** Neattext (Cleaning), TextBlob (Lemmatization), Scikit-learn (Vectorizer).
*   **Derin Öğrenme:** TensorFlow & Keras (Sequential Model).
*   **Arayüz:** Streamlit (Canlı Demo)

## 🧬 NLP İş Akışı (Pipeline)
Modelin başarısı, ham metnin aşağıdaki adımlardan titizlikle geçirilmesine dayanmaktadır:

1.  **Metin Ön İşleme (Preprocessing):**
    *   `neattext` kütüphanesi kullanılarak emojiler, özel karakterler ve çoklu boşluklar temizlenmiştir.
    *   Tüm metinler küçük harfe (lowercase) dönüştürülmüştür.
2.  **Kök Bulma (Lemmatization):**
    *   `TextBlob` ve `WordNetLemmatizer` kullanılarak kelimeler köklerine indirgenmiş, böylece modelin kelime haznesindeki gürültü azaltılmıştır.
3.  **Vektörizasyon (Sayısal Temsil):**
    *   `CountVectorizer` ile metinler sayısal matrislere dönüştürülmüştür.
    *   **N-gram (1,2):** Sadece tekil kelimelere değil, kelime çiftlerine de bakılarak LLM'lerin sık kullandığı kalıplar yakalanmıştır.
4.  **Derin Öğrenme Mimarisi:**
    *   Giriş katmanında `Dropout` kullanılarak aşırı öğrenme (overfitting) engellenmiştir.
    *   Gizli katmanlarda `ReLU` aktivasyon fonksiyonu kullanılmıştır.
    *   Çıkış katmanında `Sigmoid` fonksiyonu ile metnin % kaç olasılıkla AI olduğu hesaplanmıştır.

## 📊 Performans Sonuçları
Veri seti dış kaynaklarla dengelendikten sonra elde edilen sonuçlar:

*   **Doğruluk (Accuracy):** %99
*   **F1-Skoru (AI Sınıfı):** %99

### Karmaşıklık Matrisi (Confusion Matrix)
Modelin gerçek performansı, test setindeki insan ve AI yazılarını yüksek hassasiyetle ayırabildiğini göstermektedir.

| Gerçek / Tahmin | İnsan (0) | Yapay Zeka (1) |
| :--- | :---: | :---: |
| **İnsan (0)** | 5468 | 13 |
| **Yapay Zeka (1)** | 44 | 3449 |

## 🚀 Uygulamayı Çalıştırma
Projeyi yerel bilgisayarınızda çalıştırmak için:

1. Depoyu klonlayın: `git clone https://github.com/tugcenkrdnz/LLM_Detect_AI_Generated_Text.git`
2. Gereksinimleri yükleyin: `pip install -r requirements.txt`
3. Uygulamayı başlatın: `streamlit run app.py`

## 📂 Klasör Yapısı
*   `app.py`: Streamlit arayüz kodu.
*   `notebooks/`: Eğitim sürecini içeren `.ipynb` dosyaları.
*   `models/`: Kaydedilmiş `.keras` model ve `.pkl` vektörleştirici dosyaları.
*   `requirements.txt`: Gerekli tüm kütüphanelerin listesi.

