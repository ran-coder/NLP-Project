import joblib
from gensim.models import Word2Vec

# 1. Test TF-IDF dimensions
tfidf = joblib.load("./models/tfidf_vectorizer.pkl")
print("TF-IDF Features:", len(tfidf.get_feature_names_out())) 
# This should now print 5000!

# 2. Test Word2Vec dimensions
w2v = Word2Vec.load("./models/word2vec.model")
print("Word2Vec Vector Size:", w2v.wv.vector_size) 
# This should now print 100!

# 1. Load the Classifiers
lr_tfidf = joblib.load("./models/logistic_tfidf_model.pkl")
svm_tfidf = joblib.load("./models/svm_tfidf_model.pkl")
lr_w2v = joblib.load("./models/w2v_logistic_model.pkl")
svm_w2v = joblib.load("./models/w2v_svm_model.pkl")

print("--- TF-IDF Models Expected Features ---")
# n_features_in_ tells you exactly how many input features the model expects
print(f"LR (TF-IDF) expects:  {lr_tfidf.n_features_in_} features")
print(f"SVM (TF-IDF) expects: {svm_tfidf.n_features_in_} features")

print("\n--- Word2Vec Models Expected Features ---")
print(f"LR (Word2Vec) expects:  {lr_w2v.n_features_in_} features")
print(f"SVM (Word2Vec) expects: {svm_w2v.n_features_in_} features")