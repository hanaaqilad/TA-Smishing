# 📱 Smishing SMS Classifier

A Streamlit web app to classify SMS messages into **Penipuan (Scam)**, **Promo**, or **Normal** using a fine-tuned LLaMA 3.1 model privately hosted on Hugging Face.

---

## 💡 Overview

This project helps detect potentially malicious or promotional messages using a custom fine-tuned LLM. You can run the app either on:

- **Google Colab** (for users without local GPU)
- **Your Local Machine** (recommended if you have GPU access)

---

## 🚀 Getting Started

### Option 1: Run on Google Colab (No GPU required)

You can use the hosted notebook below to run the app directly in your browser:

👉 **[Open on Google Colab](https://colab.research.google.com/drive/1Q_KB1KJ0CvFDX3eWp4S8NyP4XEdZKHKL?usp=sharing)**

---

### Option 2: Run Locally (GPU required)

Follow the steps below if you have a GPU-enabled machine.

#### 🛠 Step 1: Clone the repository

```bash
git clone https://github.com/hanaaqilad/TA-Smishing.git
cd TA-Smishing
```

#### 🛠 Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

#### 🛠 Step 3: Create a .env file in the root directory
```bash
# .env
HUGGINGFACE_TOKEN=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
NGROK_AUTH_TOKEN=2wf3EgLOOsikEst1biP2Vm1KkbY_7eoheyS3ZRYWqnqv4E4B

```

#### 🛠 Step 4: Run the Streamlit app
```bash
streamlit run app.py
```

the app will generate a public URL and print it in the terminal. Click that generated link to visit the streamlit web.


## 📂 Project Structure

```bash
TA-Smishing/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (you create this)
└── README.md             # Project documentation
```

---

## 🔒 Security & Tokens

This project uses sensitive API tokens from:

- 🔐 **Hugging Face** (`HUGGINGFACE_TOKEN`)
- 🌐 **Ngrok** (`NGROK_AUTH_TOKEN`)

---

## 🙌 Acknowledgments

- 🤗 [Hugging Face](https://huggingface.co/)
- 🦙 LLaMA 3.1 by Meta
- 🔗 [Ngrok](https://ngrok.com/)
- 📊 [Streamlit](https://streamlit.io/) for frontend UI
