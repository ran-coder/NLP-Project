import streamlit as st
import pandas as pd
import plotly.express as px
import re
from collections import Counter
from predictors import load_all_models, predict_emotion, EMOTION_MAPPING, ORDERED_LABELS

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
    
    # Map numerical categories directly to standard string keys
    df[emotion_col] = df[emotion_col].map(EMOTION_MAPPING)
    
    # Calculate dataset metrics
    dist_df = df[emotion_col].value_counts().reset_index()
    dist_df.columns = ['Emotion', 'Count']
    
    # Basic English regex stopword removal filter for clearer analytics visualizations
    stop_words = {'the', 'and', 'a', 'of', 'to', 'is', 'in', 'i', 'that', 'it', 'you', 'on', 'for', 'with', 'was', 'this', 'my', 'have', 'but', 'as', 'about', 'am', 'so'}
    
    insights = {}
    for emotion in ORDERED_LABELS:
        emotion_df = df[df[emotion_col] == emotion]
        
        if emotion_df.empty:
            insights[emotion] = {"words": ["None Found"]}
            continue
            
        combined_text = " ".join(emotion_df[text_col].astype(str).tolist()).lower()
        
        # Tokenize words longer than 3 characters and filter generic words
        words = [w for w in re.findall(r'\b\w+\b', combined_text) if len(w) > 3 and w not in stop_words]
        common_words = [item[0] for item in Counter(words).most_common(6)]
            
        insights[emotion] = {
            "words": common_words
        }
        
    return dist_df, insights

# Initialize backend models
with st.spinner("Initializing models into memory... This may take a moment..."):
    try:
        models = load_all_models()
    except Exception as e:
        st.error(f"Failed to load backend inference models: {e}")
        st.stop()

# Build insights index 
with st.spinner("Compiling insights from local parquet engine..."):
    try:
        data_distribution, dynamic_insights = load_dataset_insights(PARQUET_FILE_PATH)
    except Exception as e:
        st.error(f"Error accessing `{PARQUET_FILE_PATH}`: {e}")
        st.info("Check that column names match your `.parquet` dataset schema exactly.")
        st.stop()

# Navigation panel setup
tab1, tab2 = st.tabs(["🔮 Real-Time Multi-Model Inference", "📊 Dataset Analytics Engine"])

# -----------------------------------------------------------------------------
# TAB 1: RUN INFERENCE PIPELINE
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Interactive Evaluation Array")
    
    user_input = st.text_area(
        "Enter target social text string for emotion classification:",
        placeholder="ENTER YOUR TEXT HERE",
        max_chars=500
    )
    
    # Simple explicit prediction emoji lookups
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
                    
                    # Look up corresponding emoji dynamically based on this specific model's output
                    pred_emoji = EMOJI_LOOKUP.get(dominant_emotion, '🧠')
                    
                    # Display the prediction along with its specific emoji
                    st.metric(
                        label="Decision Class", 
                        value=f"{pred_emoji} {str(dominant_emotion).title()}", 
                        delta=f"{dominant_conf:.1f}% Conf."
                    )
                    
                    # Plot distribution array
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

# -----------------------------------------------------------------------------
# TAB 2: METRICS VIEW
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TAB 2: METRICS VIEW
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Data Distribution & Balance Analytics")
    st.markdown(f"Direct logs compiled from storage instance: `{PARQUET_FILE_PATH}`")
    
    # Create two columns to show the Before (Original) and After (Balanced) state side-by-side
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
        
        # MIRROR YOUR EXACT NOTEBOOK LOGIC DYNAMICALLY:
        # Applies a cap of 10,000 rows per class max, matching your exact lambda expression
        CAP_VALUE = 10000  # Change this to 5000 if you update your notebook's min() argument
        
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
        
    # Informative caption summarizing your undersampling rule
    st.info(
        f"💡 **Capping Balanced Strategy:** Majority classes containing more than **{CAP_VALUE:,}** rows "
        f"are downsampled using a random state seed to prevent dominant class bias during classifier fitting."
    )