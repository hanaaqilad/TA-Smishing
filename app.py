import streamlit as st
import torch
# from unsloth import FastLanguageModel
from pyngrok import ngrok 
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
import os

# login(token=os.getenv("HUGGINGFACE_TOKEN"))
# Directly set the Hugging Face token
login(token="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")

# Konfigurasi model pake TRANSFORMERS
@st.cache_resource # agar tidak reload model terus
def load_model():
    model_id = "ilybawkugo/lora-llama3.1-8b-smishing"  # atau model kecil lain yang support CPU
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

# # Konfigurasi model pake UNSLOTH
# @st.cache_resource  # agar tidak reload model terus
# def load_model():
#     max_seq_length = 2048
#     dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#     load_in_4bit = True
#     model_name = "ilybawkugo/lora-llama3.1-8b-smishing"


#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = model_name,
#         max_seq_length = max_seq_length,
#         dtype = dtype,
#         load_in_4bit = load_in_4bit,
#         device_map = "auto",
#         trust_remote_code = True,
#     )
#     FastLanguageModel.for_inference(model)
#     return model, tokenizer

model, tokenizer = load_model()

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

# Fungsi klasifikasi
def classify_sms(sms):
    prompt = prompt_cot.format(sms, "")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            temperature=0.0,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:")[-1].strip().split("###")[0].strip().split("\n")[0]
    return response

# UI Streamlit
st.title("🚨 Klasifikasi SMS: Penipuan, Promo, atau Normal")
st.markdown("Masukkan isi SMS dan model akan menganalisisnya.")

sms_input = st.text_area("Isi SMS", height=150)

if st.button("Klasifikasikan"):
    with st.spinner("Menganalisis..."):
        result = classify_sms(sms_input)
        st.success(f"Hasil Prediksi: **{result}**")


# Expose Streamlit app to the web using ngrok
public_url = ngrok.connect(8501)

# Run Streamlit app
os.system(f"streamlit run app.py &")

# Print the public URL
print(f"Streamlit app is running at: {public_url}")

