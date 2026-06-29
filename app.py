import streamlit as st
import pandas as pd
import plotly.express as px
import re
import string
from collections import Counter

# Import NLTK processing utilities to match your training notebook pipeline exactly
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from predictors import load_all_models, predict_emotion, EMOTION_MAPPING, ORDERED_LABELS

# Pre-download required NLTK resources once on app initialization
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk_resources()

# Application styling adjustments
st.set_page_config(page_title="Social Media Emotion Analyzer", layout="wide")

st.title("🧠 Social Media Emotion Analyzer")
st.markdown("Compare an operational fine-tuned **BERT** pipeline against **4 Classic Machine Learning configurations**.")

PARQUET_FILE_PATH = './data/train-00000-of-00001.parquet'

@st.cache_data
def load_dataset_insights(file_path):
    """Processes your local parquet split to calculate distribution metrics and clean keywords."""
    df = pd.read_parquet(file_path)
    
    text_col = 'text'       
    emotion_col = 'label'   
    
    df[emotion_col] = df[emotion_col].map(EMOTION_MAPPING)
    
    # Calculate dataset metrics
    dist_df = df[emotion_col].value_counts().reset_index()
    dist_df.columns = ['Emotion', 'Count']
    
    stop_words = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    
    insights = {}
    for emotion in ORDERED_LABELS:
        emotion_df = df[df[emotion_col] == emotion]
        
        if emotion_df.empty:
            insights[emotion] = {"words": ["None Found"]}
            continue
            
        corpus = emotion_df[text_col].astype(str).tolist()
        
        lower_corpus = [i.lower() for i in corpus]
        words = [word_tokenize(i) for i in lower_corpus]
        
        cleaned_tokens_pool = []
        for sentence in words:
            for token in sentence:
                cleaned_token = token.strip(string.punctuation)
                if cleaned_token and cleaned_token not in stop_words and cleaned_token.isalpha():
                    lemmatized = lem.lemmatize(cleaned_token)
                    cleaned_tokens_pool.append(lemmatized)
        
        common_words = [item[0] for item in Counter(cleaned_tokens_pool).most_common(6)]
            
        insights[emotion] = {
            "words": common_words if common_words else ["No words extracted"]
        }
        
    return dist_df, insights

with st.spinner("Initializing models into memory... This may take a moment..."):
    try:
        models = load_all_models()
    except Exception as e:
        st.error(f"Failed to load backend inference models: {e}")
        st.stop()

with st.spinner("Compiling insights from local parquet engine..."):
    try:
        data_distribution, dynamic_insights = load_dataset_insights(PARQUET_FILE_PATH)
    except Exception as e:
        st.error(f"Error accessing `{PARQUET_FILE_PATH}`: {e}")
        st.info("Check that column names match your `.parquet` dataset schema exactly.")
        st.stop()

tab1, tab2 = st.tabs(["🔮 Real-Time Multi-Model Inference", "📊 Dataset Analytics Engine"])

with tab1:
    st.subheader("Interactive Evaluation Array")
    
    user_input = st.text_area(
        "Enter target social text string for vector evaluation:",
        placeholder="Enter your social text here.",
        max_chars=500
    )
    
    EMOJI_LOOKUP = {
        'sadness': '😢',
        'joy': '😊',
        'love': '🥰',
        'anger': '😠',
        'fear': '😨',
        'surprise': '😲'
    }
    
    if st.button("Execute Vector Classifications", type="primary"):
        if not user_input.strip():
            st.warning("Input area cannot be processing blank strings. Write text first.")
        else:
            cols = st.columns(len(models))
            
            for idx, (name, model_info) in enumerate(models.items()):
                with cols[idx]:
                    st.markdown(f"##### **{name}**")
                    
                    predictions = predict_emotion(user_input, name, model_info)
                    
                    if not predictions:
                        st.error("Inference Error")
                        continue
                    
                    dominant_emotion = max(predictions, key=predictions.get)
                    dominant_conf = predictions[dominant_emotion] * 100
                    
                    pred_emoji = EMOJI_LOOKUP.get(dominant_emotion, '🧠')
                    
                    st.metric(
                        label="Decision Class", 
                        value=f"{pred_emoji} {str(dominant_emotion).title()}", 
                        delta=f"{dominant_conf:.1f}% Conf."
                    )
                    
                    df_pred = pd.DataFrame(list(predictions.items()), columns=["Emotion", "Confidence"])
                    
                    fig = px.bar(
                        df_pred, 
                        x="Confidence", 
                        y="Emotion", 
                        orientation='h', 
                        color="Emotion", 
                        category_orders={"Emotion": ORDERED_LABELS}
                    )
                    fig.update_layout(showlegend=False, height=200, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_{name}")

with tab2:
    st.subheader("Data Distribution & Balance Analytics")
    # st.markdown(f"Direct logs compiled from storage instance: `{PARQUET_FILE_PATH}`")
    
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.markdown("### **⚖️ Before: Original Distribution**")
        fig_before = px.bar(
            data_distribution, 
            x='Emotion', 
            y='Count', 
            color='Emotion', 
            text_auto=True, 
            category_orders={"Emotion": ORDERED_LABELS},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_before.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Emotion Class",
            yaxis_title="Number of Samples"
        )
        st.plotly_chart(fig_before, use_container_width=True, key="chart_before_balance")

    with col_graph2:
        st.markdown("### **🔄 After: Balanced Distribution (Capped)**")
        
        CAP_VALUE = 8000  
        
        balanced_distribution = data_distribution.copy()
        balanced_distribution['Count'] = balanced_distribution['Count'].apply(lambda x: min(x, CAP_VALUE))
        
        fig_after = px.bar(
            balanced_distribution, 
            x='Emotion', 
            y='Count', 
            color='Emotion', 
            text_auto=True, 
            category_orders={"Emotion": ORDERED_LABELS},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_after.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Emotion Class",
            yaxis_title="Number of Samples"
        )
        st.plotly_chart(fig_after, use_container_width=True, key="chart_after_balance")
        
    st.info(
        f"💡 **Capping Balanced Strategy:** Majority classes containing more than **{CAP_VALUE:,}** rows "
        f"are downsampled using a random state seed to prevent dominant class bias during classifier fitting."
    )

    st.markdown("---")

    st.markdown("### **🔍 Linguistic Feature Explorer**")
    st.markdown("Select a class category below to inspect top tokens parsed using your definitive preprocessing notebook rules:")
    
    selected_emotion = st.selectbox("Select target class category:", ORDERED_LABELS, key="insights_selector")
    
    st.write(f"**Top Feature Associated Keywords for** `{selected_emotion}`:")
    
    if selected_emotion in dynamic_insights and "words" in dynamic_insights[selected_emotion]:
        keywords_list = dynamic_insights[selected_emotion]["words"]
        st.markdown(" ".join([f"`{word}`" for word in keywords_list]))
    else:
        st.markdown("`No words extracted for this subset.`")