# 🎭 Emotion Analyzer — NLP Sentiment & Emotion Classification

A comparative NLP project that deploys emotion classification using both traditional Machine Learning models (TF-IDF, Word2Vec) and an advanced transformer-based model (BERT), trained on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset.

---

## 📌 Project Overview

This project explores and compares multiple feature extraction and classification approaches for 6-class emotion detection:

| Emotion | Label |
|---------|-------|
| Sadness | 0 |
| Joy | 1 |
| Love | 2 |
| Anger | 3 |
| Fear | 4 |
| Surprise | 5 |

**Models implemented:**

| Extractor | Classifier 
|-----------|-----------
| TF-IDF | SVM (LinearSVC) 
| TF-IDF | Logistic Regression 
| Word2Vec (avg pooling) | SVM 
| Word2Vec (avg pooling) | LR 
| BERT (`roberta-base-go_emotions`) | Fine-tuned Transformer


## ⚙️ Setup & Installation

Use Python Version 3.12
### 1. Clone the repository

```bash
git clone https://github.com/ran-coder/nlp_project.git
cd nlp_project
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

The raw dataset is not included in this repository. Download it from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("dair-ai/emotion")
```
---

## 🚀 Run Order

> **Important:** Run notebooks in this exact order to regenerate all models and feature files.

```
1. preprocess.ipynb       ← clean, balance, and save balanced_data.csv
2. ml_model.ipynb         ← train TF-IDF + SVM/LR models, Word2Vec + SVM/LR models
4. bert_model.ipynb        ← fine-tune BERT transformer
```

Make sure to saves all the outputs to `models/` .

---

## 🤖 Model Details

### Traditional ML Models (TF-IDF & Word2Vec)

**TF-IDF + SVM / LR**
- Vectorizer: `TfidfVectorizer(max_features=5000, norm='l2')`
- Captures word importance and discriminative features
- Strong baseline; interpretable feature weights
- Fast to train and predict

**Word2Vec + SVM / LR**
- 100-dimensional embeddings trained on the emotion corpus
- Document vector = mean of word vectors
- Captures semantic similarity between words
- Average pooling may dilute strong sentiment signals

### Advanced Model (BERT Transformer)

**`SamLowe/roberta-base-go_emotions`**
- Pre-trained transformer fine-tuned on GoEmotions dataset
- Ekman-collapsed to 6 emotion categories
- Captures contextual meaning, negation, and nuance
- Highest accuracy; requires GPU for optimal training speed

---

## 📊 Results Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| TF-IDF + SVM | — | Strong baseline |
| TF-IDF + LR | — | Most interpretable |
| Word2Vec + SVM | — | Semantic-aware |
| Word2Vec + LR | — | Semantic-aware |
| BERT | — | Best overall |


---

## 🌐 Deployment

The project is deployed as an interactive web app using **Streamlit**.

```bash
streamlit run app.py
```

The app allows users to:
- Type any text input
- View the predicted emotion with confidence scores
- Compare predictions across all models side by side


## 📄 License

This project is for academic purposes.
