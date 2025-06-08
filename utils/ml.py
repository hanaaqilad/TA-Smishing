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

### Supporting function ###
def identity(x): 
    return x

### Load model ML ###
def load_ml_assets():
    with open("utils/svm.pkl", "rb") as f:
        model_ml = pickle.load(f)
    with open("utils/tf_idf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("utils/minmax_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model_ml, vectorizer, scaler

### Predict with ML model ###
def predict_ml(model_ml, final_input):
    pred = model_ml.predict(final_input)[0]
    if pred == 0:
        return "normal"
    elif pred == 1:
        return "penipuan"
    else:
        return "promo"

### Feature Engineering ###
def extract_features(df, text_column):
    def has_phone_num(text):
        reNoTelp = r'\+?(?:\d[\s\-]?){8,14}'
        return 1 if re.search(reNoTelp, text, re.IGNORECASE) else 0

    def has_url(text):
        reUrl = r"https?[;:/]{1,3}[^\s\"\'<>]+|www\.[^\s\"\'<>]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\/[^\s\"\'<>]+"
        return 1 if re.search(reUrl, text, re.IGNORECASE) else 0

    def extract_url(text):
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

### Cleaning and Standardize SMS input ###
def clean_and_standardize(text):
    # preparation
    nltk.download('punkt_tab')
    nltk.download('stopwords')

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Gabung lagi jadi string
    return ' '.join(tokens)

### Hanlde overall ML processes ###
def handle_ml(sms, vectorizer, scaler):
    df = pd.DataFrame([{
        'teks': sms
    }])

    df = extract_features(df, text_column="teks") 
    df['teks_standardized'] = df['teks'].apply(clean_and_standardize)
       
    X_text_standardized = df['teks_standardized']
    X_text_tfidf = vectorizer.transform(X_text_standardized)

    # for scaler handling
    num_cols = ['cnt_enter', 'alphanum_ratio', 'cnt_special_chars', 'all_caps_ratio', 'symbol_ratio']
    bin_cols = ['phone_num', 'has_url', 'has_money']
    def identity(x): 
        return x
    
    X_numerik = df.drop(columns=['url_text','teks_standardized','has_email','teks'])
    X_numerik_scaled = scaler.transform(X_numerik)

    X = hstack([X_text_tfidf, X_numerik_scaled])

    return X
