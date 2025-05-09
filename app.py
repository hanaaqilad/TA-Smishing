import streamlit as st
import torch
# from unsloth import FastLanguageModel
# from pyngrok import ngrok 
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from huggingface_hub import login
import os

# ML imports
import re
import nltk
import string
import pickle
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import hstack


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# set Hugging Face token and authenticate 
load_dotenv() 
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

### Load model ML ###
def load_ml_assets():
    with open("svm_model.pkl", "rb") as f:
        model_ml = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model_ml, vectorizer

model_ml, vectorizer = load_ml_assets()

nltk.download('punkt_tab')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Konfigurasi model pake TRANSFORMERS
@st.cache_resource # agar tidak reload model terus
def load_llm():
    model_id = "ilybawkugo/lora-llama3.1-8b-smishing"  # atau model kecil lain yang support CPU
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    return llm, tokenizer

# # Konfigurasi model pake UNSLOTH
# @st.cache_resource  # agar tidak reload model terus
# def load_llm():
#     max_seq_length = 2048
#     dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#     load_in_4bit = True
#     model_name = "ilybawkugo/lora-llama3.1-8b-smishing"
#     llm, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = model_name,
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#         device_map = "auto",
#         trust_remote_code = True,
#     )
#     FastLanguageModel.for_inference(llm)
#     return llm, tokenizer

llm, tokenizer = load_llm()

# Prompt template
prompt_cot = """
  Berikut adalah sebuah instruksi yang menjelaskan sebuah tugas, diikuti dengan sebuah input yang memberikan konteks tambahan. Tulislah respons yang sesuai untuk menyelesaikan permintaan tersebut.

  ### Instruction:
  Baca pesan tersebut lalu ikuti langkah-langkah berikut.
  Langkah 1: Apakah pesan ini mengandung tautan/link?
  Langkah 2: Jika pesan ini mengandung tautan/link, apakah tautan/link tersebut sah atau mencurigakan?
  Langkah 3: Apakah pesan ini mengandung nomor telepon?
  Langkah 4: Apakah pesan ini mengandung alamat email?
  Langkah 5: Apakah pesan ini mengandung indikasi kalimat terkait uang?
  Langkah 6: Apakah pesan ini mengandung indikasi hadiah?
  Langkah 7: Apakah pesan ini mengandung simbol aneh?
  Langkah 8: Apakah pesan ini mengandung huruf dengan case yang tidak beraturan?
  Langkah 9: Berdasarkan langkah-langkah tersebut, tentukan apakah ini penipuan, promo, atau normal?
  Jawab Penipuan/Promo/Normal

  ### Input:
  {}

  ### Response:
  {}
"""

# Fungsi klasifikasi GEN AI
def classify_sms(sms):
    prompt = prompt_cot.format(sms, "")
    inputs = tokenizer([prompt], return_tensors="pt").to(llm.device)

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            temperature=0.0,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip().split("###")[0].strip().split("\n")[0]
    return response


# authenticate 
# def preprocess_sms(text):
#     # 1. Lowercase, remove symbol, dsb.
#     # 2. Extract fitur baru (e.g., capital ratio, money terms, etc.)
#     # 3. Apply TF-IDF
#     # 4. Combine TF-IDF + fitur buatan → jadi input final

#     tfidf_vec = tfidf.transform([text])
#     # gabungkan fitur tambahan, misalnya scaler.fit_transform([features])
#     final_input = tfidf_vec  # atau pakai hstack
#     return final_input


# ML
def handle_ml(sms):
    df = pd.DataFrame([{
        'teks': sms
    }])

    # Numerical feature engineering for ML
    def extract_features(df, text_column):
        def has_no_telp(text):
            reNoTelp = r'\+?(?:\d[\s\-]?){8,14}'
            return 1 if re.search(reNoTelp, text, re.IGNORECASE) else 0

        # def has_url(text):
        #   reUrl = r'(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?'
        #   return 1 if re.search(reUrl, text, re.IGNORECASE) else 0

        # def extract_url(text):
        #   reUrl = r'(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?'
        #   return re.findall(reUrl, text, re.IGNORECASE)

        def has_url(text):
            # reUrl = r"https?://[^\s\"\'<>]+|www\.[^\s\"\'<>]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\/[^\s\"\'<>]+"
            reUrl = r"https?[;:/]{1,3}[^\s\"\'<>]+|www\.[^\s\"\'<>]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\/[^\s\"\'<>]+"
            return 1 if re.search(reUrl, text, re.IGNORECASE) else 0

        def extract_url(text):
            # reUrl = r"https?://[^\s\"\'<>]+|www\.[^\s\"\'<>]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\/[^\s\"\'<>]+"
            reUrl = r"https?[;:/]{1,3}[^\s\"\'<>]+|www\.[^\s\"\'<>]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\/[^\s\"\'<>]+"
            return re.findall(reUrl, text, re.IGNORECASE)

        def has_email(text):
            reEmail = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            return 1 if re.search(reEmail, text, re.IGNORECASE) else 0

        def has_money_terms(text):
            money_terms = ['uang', 'dana', 'jt', 'juta', 'rb', 'rp', 'ribu', 'rupiah', 'milyar', 'miliar', 'triliun', 'trilyun', '%', 'modal']
            return int(any(term in text.lower() for term in money_terms))

        def num_linebreaks(text):
            return text.count('\n')

        def alphanum_word_ratio(text):
            words = text.split()
            if not words:
                return 0
            alphanum = [w for w in words if re.search(r'[0-9][A-Za-z]|[A-Za-z][0-9]', w, re.IGNORECASE)]
            return len(alphanum) / len(words)

        def count_special_chars(text):
            pattern = r"[@!'\"*\$%\^&\(\)=_+~`\[\]\{\}<>\|\\\/#,:;]"
            return len(re.findall(pattern, text))

        def all_caps_ratio(text):
            words = text.split()
            if not words:
                return 0
            return sum(1 for w in words if w.isupper()) / len(words)

        def symbol_word_ratio(text):
            words = text.split()
            if not words:
                return 0
            symbol_words = [w for w in words if re.search(r"[A-Za-z]+[^A-Za-z0-9\s]+[A-Za-z]+", w, re.IGNORECASE)]
            return len(symbol_words) / len(words)

        # Apply all feature functions
        df['no_telp'] = df[text_column].apply(has_no_telp)
        df['url'] = df[text_column].apply(has_url)
        df['url_text'] = df[text_column].apply(extract_url)
        df['has_url'] = df['url_text'].apply(lambda x: 1 if len(x) > 0 else 0)
        df['email'] = df[text_column].apply(has_email)
        df['has_money'] = df[text_column].apply(has_money_terms)
        df['cnt_enter'] = df[text_column].apply(num_linebreaks)
        df['alphanum_ratio'] = df[text_column].apply(alphanum_word_ratio)
        df['cnt_special_chars'] = df[text_column].apply(count_special_chars)
        df['all_caps_ratio'] = df[text_column].apply(all_caps_ratio)
        df['symbol_ratio'] = df[text_column].apply(symbol_word_ratio)

        return df


    # create function for cleaning and standardize the texts
    def clean_and_standardize(text):
        # 1. Lowercase
        text = text.lower()

        # 2. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 3. Tokenize
        tokens = word_tokenize(text)

        # 4. Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # # 5. Stemming
        # stemmed = [stemmer.stem(word) for word in tokens]

        # 6. Gabung lagi jadi string
        return ' '.join(tokens)

    df = extract_features(df, text_column="teks") 
    df['teks_standardized'] = df['teks'].apply(clean_and_standardize)
    X_numerik = df.drop(columns=['label','url_text','teks_standardized', 'teks'])
    X_text_standardized = df['teks_standardized']
    X_text_tfidf = vectorizer.transform(X_text_standardized)
    X = hstack([X_text_tfidf, X_numerik])

    return X


def predict_ml(final_input):
    pred = model_ml.predict(final_input)[0]
    if pred == 0:
        return "normal"
    elif pred == 1:
        return "penipuan"
    else:
        return "promo"


sms_input = st.text_area("Masukkan isi SMS")
if st.button("Prediksi ML"):
    final_input = handle_ml(sms_input)
    result_ml = predict_ml(final_input)
    st.success(f"Hasil prediksi: **{result_ml}**")


# UI Streamlit
st.title("🚨 Klasifikasi SMS: Penipuan, Promo, atau Normal")
st.markdown("Masukkan isi SMS dan model akan menganalisisnya.")

sms_input = st.text_area("Isi SMS", height=150)

# if st.button("Cek", type="primary"):
#     with st.spinner("Menganalisis..."):
#         result = classify_sms(sms_input)
#         if result == 'penipuan':
#             st.error(f"Hasil prediksi: **Penipuan**")
#         elif result == 'promo':
#             st.warning(f"Hasil prediksi: **Promo**")
#         else:
#             st.success(f"Hasil prediksi: **Normal**")

# === TOMBOL PREDIKSI ===
if st.button("Cek", type="primary"):
    with st.spinner("Menganalisis..."):

        # === Prediksi ML ===
        final_input = handle_ml(sms_input)         
        result_ml = predict_ml(final_input)  

        # === Prediksi GenAI ===
        result_genai = classify_sms(sms_input)  

        # === Tampilkan Hasil ===
        st.markdown("### 🔎 Hasil Prediksi Machine Learning:")
        if result_ml.lower() == 'penipuan':
            st.error(f"**{result_ml}**")
        elif result_ml.lower() == 'promo':
            st.warning(f"**{result_ml}**")
        else:
            st.success(f"**{result_ml}**")

        st.markdown("### 🤖 Hasil Prediksi Generative AI:")
        if result_genai.lower() == 'penipuan':
            st.error(f"**{result_genai}**")
        elif result_genai.lower() == 'promo':
            st.warning(f"**{result_genai}**")
        else:
            st.success(f"**{result_genai}**")


# Expose Streamlit app to the web using ngrok
# public_url = ngrok.connect(8501)

# Run Streamlit app
os.system(f"streamlit run app.py &")

# Print the public URL
# print(f"Streamlit app is running at: {public_url}")

