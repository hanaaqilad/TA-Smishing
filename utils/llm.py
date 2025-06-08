import os
import torch
import streamlit as st
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
### import unsloth
# from pyngrok import ngrok 


### Auth HuggingFace Token ###
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HF_TOKEN)

### Best Prompt ###
BEST_PROMPT = """
    Berikut adalah sebuah instruksi yang menjelaskan sebuah tugas, diikuti dengan sebuah input yang memberikan konteks tambahan. Tulislah respons yang sesuai untuk menyelesaikan permintaan tersebut.

    ### Instruction:
    Tentukan apakah teks berikut merupakan pesan penipuan, pesan promo, atau pesan normal. Jawab dengan hanya menggunakan satu kata (Penipuan/Promo/Normal).

    ### Input:
    {}

    ### Response:
    {}
"""

### Load LLM ###
@st.cache_resource # agar tidak reload model terus
def load_llm():
    try:
        from unsloth import FastLanguageModel
        UNSLOTH_AVAILABLE = True
    except ImportError:
        UNSLOTH_AVAILABLE = False

    model_id = "ilybawkugo/lora_lama_2e-4-48-1024"
    max_seq_length = 1024
    epoch = "epoch-5"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        if UNSLOTH_AVAILABLE:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = model_id,
                revision = epoch,                   
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
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        return model, tokenizer

### GenAI Classifier ###
def classify_sms(sms, llm, tokenizer):
    prompt = BEST_PROMPT.format(sms, "")
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
