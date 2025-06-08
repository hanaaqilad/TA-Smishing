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
st.title("🚨 Klasifikasi SMS: Penipuan, Promo, atau Normal")
st.markdown("Masukkan isi SMS dan model akan menganalisisnya.")

### Input SMS ###
sms_input = st.text_area("Isi SMS", height=150)

### UI Website ###
if st.button("Cek", type="primary"):
    with st.spinner("Menganalisis..."):

        # Prediksi ML
        final_input = handle_ml(sms_input, vectorizer, scaler)
        result_ml = predict_ml(model_ml, final_input)

        # Prediksi Gen AI
        result_genai = classify_sms(sms_input, llm, tokenizer)

        # Tampilkan Hasil
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

# Run Streamlit app
os.system(f"streamlit run app.py &")