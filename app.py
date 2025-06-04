import streamlit as st
import torch
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
    with open("svm.pkl", "rb") as f:
        model_ml = pickle.load(f)
    with open("tf_idf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model_ml, vectorizer, scaler

model_ml, vectorizer, scaler = load_ml_assets()

nltk.download('punkt_tab')
nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

### import unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

@st.cache_resource # agar tidak reload model terus
def load_llm():
    model_id = "ilybawkugo/lora_lama_2e-4-48-1024"
    max_seq_length = 1024
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        if UNSLOTH_AVAILABLE:
            # st.info("🔁 Loading model using **Unsloth** backend...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_id,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = True,
                device_map = "auto",
                trust_remote_code = True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        else:
            raise ImportError("Unsloth not installed.")
        
    # Fallback ke Hugging Face Transformers
    except Exception as e:
        # st.warning(f"⚠️ Unsloth failed: {e}\n\nFalling back to Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        return model, tokenizer

llm, tokenizer = load_llm()

# Prompt template
best_prompt = """
    Berikut adalah sebuah instruksi yang menjelaskan sebuah tugas, diikuti dengan sebuah input yang memberikan konteks tambahan. Tulislah respons yang sesuai untuk menyelesaikan permintaan tersebut.

    ### Instruction:
    Tentukan apakah teks berikut merupakan pesan penipuan, pesan promo, atau pesan normal. Jawab dengan hanya menggunakan satu kata (Penipuan/Promo/Normal).

    ### Input:
    {}

    ### Response:
    {}
"""

# Fungsi klasifikasi GEN AI
def classify_sms(sms):
    prompt = best_prompt.format(sms, "")
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
        def has_phone_num(text):
            reNoTelp = r'\+?(?:\d[\s\-]?){8,14}'
            return 1 if re.search(reNoTelp, text, re.IGNORECASE) else 0

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
            money_terms = ['uang', 'dana', 'jt', 'juta', 'rb', 'rp', 'ribu', 'ratus', 'rupiah', 'milyar', 'miliar', 'triliun', 'trilyun', '%', 'modal']
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
        df['phone_num'] = df[text_column].apply(has_phone_num)
        df['has_url'] = df[text_column].apply(has_url)
        df['url_text'] = df[text_column].apply(extract_url)
        df['has_email'] = df[text_column].apply(has_email)
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
    X_numerik = df.drop(columns=['label','url_text','teks_standardized','has_email','teks'])
    X_text_standardized = df['teks_standardized']
    
    X_text_tfidf = vectorizer.transform(X_text_standardized)
    X_numerik_scaled = scaler.transform(X_numerik)

    X = hstack([X_text_tfidf, X_numerik_scaled])

    return X


def predict_ml(final_input):
    pred = model_ml.predict(final_input)[0]
    if pred == 0:
        return "normal"
    elif pred == 1:
        return "penipuan"
    else:
        return "promo"


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

