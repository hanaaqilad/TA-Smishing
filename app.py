import streamlit as st
import os

from utils.ml import load_ml_assets, handle_ml, predict_ml, identity
from utils.llm import load_llm, classify_sms

from dotenv import load_dotenv
load_dotenv()

### Load models ###
model_ml, vectorizer, scaler = load_ml_assets()
llm, tokenizer = load_llm()

### Streamlit ###
st.header("🚨 Klasifikasi SMS: ")
st.subheader("Penipuan, Promo, atau Normal", divider="gray")
st.markdown("Masukkan isi SMS dan model akan menganalisisnya.")

### Input SMS ###
sms_input = st.text_area("Input isi SMS", height=150)

col1, col2 = st.columns([3, 1])
with col1:
    sms_input = st.text_area("Isi SMS", height=150)

with col2:
    cek = st.button("Cek", type="primary", use_container_width=True)

### Hasil ###
if cek and sms_input.strip():
    with st.spinner("Menganalisis..."):

        # Prediksi
        final_input = handle_ml(sms_input, vectorizer, scaler)
        result_ml = predict_ml(model_ml, final_input)
        result_genai = classify_sms(sms_input, llm, tokenizer)

        # Tampilkan Hasil
        col_ml, col_gen = st.columns(2)

        with col_ml:
            st.markdown("### ⚙️ Machine Learning")
            if result_ml.lower() == 'penipuan':
                st.error(f"**{result_ml}**")
            elif result_ml.lower() == 'promo':
                st.warning(f"**{result_ml}**")
            else:
                st.success(f"**{result_ml}**")

        with col_gen:
            st.markdown("### 🧠 Generative AI")
            if result_genai.lower() == 'penipuan':
                st.error(f"**{result_genai}**")
            elif result_genai.lower() == 'promo':
                st.warning(f"**{result_genai}**")
            else:
                st.success(f"**{result_genai}**")

# Run Streamlit app
os.system(f"streamlit run app.py &")