from flask import Flask, request, jsonify , render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords


model = load_model(r"D:\ITI\NLP\project\lstm_model.h5")
with open(r"D:\ITI\NLP\project\tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


nltk.download('stopwords')
# define the Arabic stop words list
ar_stopwords = set(stopwords.words('arabic'))

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

url_pattern = re.compile(r'https?://\S+')
# english_pattern = re.compile("[a-zA-Z]+")
arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                            ـ    | # Tatwil/Kashida
                         """, re.VERBOSE)


def tokenization(text):
  seq = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(seq, maxlen=250)
  return padded

def preprocess_input(text):
    text = re.sub(r'@\w+', '', text)            
    text = re.sub(r'\d+', '', text)             
    text = text.replace('\n', ' ')
    text = emoji_pattern.sub(r'', text)
    text = url_pattern.sub(r'', text)
    words = text.split()
    #filtered_words = [word for word in words if word not in ar_stopwords]
    #text = ' '.join(filtered_words)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)    
    text = re.sub(r'\s+', ' ', text)              
    text = ''.join([c for c in text if c not in string.punctuation])
    text = re.sub(r'؟', ' ', text) 
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub("چ", "ج", text)
    text = re.sub("ژ", "ز", text)
    text = re.sub("پ", "ب", text)
    text = re.sub(arabic_diacritics, '', text)
    print(text)
    tokens = tokenization(text)
    return tokens

def transform_number_to_text(number):
    number_to_text = {
        0: "مصري",
        1: "لبناني",
        2: "ليبي",
        3: "مغربي",
        4: "سعودي"
    }

    return number_to_text.get(number)


# Convert the predictions to a format that can be returned as JSON

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    
    text = request.form.get('text')
    input_data = preprocess_input(text)

    # Use the model to make predictions
    predictions = model.predict(input_data)
    #print(predictions)
    predictions = np.argmax(predictions,axis = 1)
    predictions = transform_number_to_text(predictions[0])

    return render_template("index.html" , predictions_text = f"result : {predictions}")
if __name__ == '__main__':
    app.run(debug=True)