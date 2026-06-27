import streamlit as st
import torch
import joblib
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.models import Word2Vec
import re

EMOTION_MAPPING = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
ORDERED_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

@st.cache_resource
def load_all_models():
    """Loads tokenizer, BERT, TF-IDF vectorizer, and 4 ML models into memory once."""
    bert_path = "./models/bert" 
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    bert_model.eval()  
    
    vectorizer = joblib.load("./models/tfidf_vectorizer.pkl")
    w2v_model = Word2Vec.load("models/word2vec.model")
    
    lr_tfidf = joblib.load("./models/logistic_tfidf_model.pkl")
    svm_tfidf = joblib.load("./models/svm_tfidf_model.pkl")
    lr_w2v = joblib.load("./models/w2v_logistic_model.pkl")
    svm_w2v = joblib.load("./models/w2v_svm_model.pkl")
    
    models = {
        "BERT (Transformer)": {"model": bert_model, "tokenizer": tokenizer, "type": "dl"},
        "LR (TF-IDF)": {"model": lr_tfidf, "extractor": vectorizer, "type": "ml_tfidf"},
        "SVM (TF-IDF)": {"model": svm_tfidf, "extractor": vectorizer, "type": "ml_tfidf"},
        "LR (Word2Vec)": {"model": lr_w2v, "extractor": w2v_model, "type": "ml_w2v"},
        "SVM (Word2Vec)": {"model": svm_w2v, "extractor": w2v_model, "type": "ml_w2v"}
    }
    return models

def get_word2vec_average(text, w2v_model):
    """Tokenizes string and calculates average word vector embedding for the document."""
    words = re.findall(r'\b\w+\b', text.lower())
    vector_size = w2v_model.wv.vector_size
    
    vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    
    if len(vectors) == 0:
        return np.zeros((1, vector_size))
    
    return np.mean(vectors, axis=0).reshape(1, -1)

def predict_emotion(text, model_name, model_info):
    """Inference router handling deep learning, sparse TF-IDF strings, or dense Word2Vec matrices."""
    probabilities = []
    model = model_info["model"]
    emergency_fallback = dict(zip(ORDERED_LABELS, [0.0] * len(ORDERED_LABELS)))
    
    try:
        # --- Category A: Deep Learning BERT ---
        if model_info["type"] == "dl":
            tokenizer = model_info["tokenizer"]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
            prob_map = {EMOTION_MAPPING[i]: probs[i] for i in range(len(probs))}
            probabilities = [prob_map[label] for label in ORDERED_LABELS]
            
        # --- Category B: ML Classifier with TF-IDF Vectorizer ---
        elif model_info["type"] == "ml_tfidf":
            vectorizer = model_info["extractor"]
            features = vectorizer.transform([text])
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                prob_map = {}
                for idx, c in enumerate(model.classes_):
                    try:
                        class_key = EMOTION_MAPPING[int(c)]
                    except (ValueError, TypeError, KeyError):
                        class_key = str(c).lower().strip()
                    prob_map[class_key] = probs[idx]
                probabilities = [prob_map.get(label, 0.0) for label in ORDERED_LABELS]
            else:  
                pred = model.predict(features)[0]
                try:
                    predicted_label = EMOTION_MAPPING[int(pred)]
                except (ValueError, TypeError, KeyError):
                    predicted_label = str(pred).lower().strip()
                probabilities = [1.0 if label == predicted_label else 0.0 for label in ORDERED_LABELS]

        # --- Category C: ML Classifier with Word2Vec Mean Embeddings ---
        elif model_info["type"] == "ml_w2v":
            w2v_model = model_info["extractor"]
            features = get_word2vec_average(text, w2v_model)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                prob_map = {}
                for idx, c in enumerate(model.classes_):
                    try:
                        class_key = EMOTION_MAPPING[int(c)]
                    except (ValueError, TypeError, KeyError):
                        class_key = str(c).lower().strip()
                    prob_map[class_key] = probs[idx]
                probabilities = [prob_map.get(label, 0.0) for label in ORDERED_LABELS]
            else:
                pred = model.predict(features)[0]
                try:
                    predicted_label = EMOTION_MAPPING[int(pred)]
                except (ValueError, TypeError, KeyError):
                    predicted_label = str(pred).lower().strip()
                probabilities = [1.0 if label == predicted_label else 0.0 for label in ORDERED_LABELS]
                
        return dict(zip(ORDERED_LABELS, probabilities))

    except Exception as e:
        print(f"Exception encountered inside predictor thread for {model_name}: {e}")
        return emergency_fallback