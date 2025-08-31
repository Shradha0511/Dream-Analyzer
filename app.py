import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as pltj
import os
from PIL import Image
import json
import re
from io import BytesIO
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Dream Journal & Analysis",
    page_icon="🌙",
    layout="wide",
)

# --- MODEL LOADING FUNCTIONS ---

@st.cache_resource
def load_models():
    """Load all the trained models and encoders"""
    models = {}
    
    # Load encoders
    models["label_encoder"] = joblib.load("label_encoder.pkl")
    models["word2idx"] = joblib.load("word2idx.pkl")
    
    # Load TFIDF
    models["tfidf"] = joblib.load("tfidf.pkl")
    
    # Load trained models
    models["nb_pipeline"] = joblib.load("nb_pipeline.pkl")
    models["lgb_model"] = joblib.load("lgb_model.pkl")
    models["log_blender"] = joblib.load("log_blender.pkl")
    
    # Load CNN model
    vocab_size = len(models["word2idx"])
    embed_dim = 128
    num_classes = len(models["label_encoder"].classes_)
    pad_idx = models["word2idx"]["<PAD>"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define CNN model architecture
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes, pad_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
            self.conv1 = nn.Conv1d(embed_dim, 100, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(embed_dim, 100, kernel_size=4, padding=2)
            self.conv3 = nn.Conv1d(embed_dim, 100, kernel_size=5, padding=2)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(300, num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = x.transpose(1, 2)
            x1 = torch.relu(self.conv1(x))
            x2 = torch.relu(self.conv2(x))
            x3 = torch.relu(self.conv3(x))
            x1 = torch.max(x1, dim=2)[0]
            x2 = torch.max(x2, dim=2)[0]
            x3 = torch.max(x3, dim=2)[0]
            x = torch.cat((x1, x2, x3), dim=1)
            x = self.dropout(x)
            return self.fc(x)
    
    # Initialize model
    cnn_model = TextCNN(vocab_size, embed_dim, num_classes, pad_idx).to(device)
    
    # Load trained weights
    cnn_model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
    cnn_model.eval()
    
    models["cnn_model"] = cnn_model
    models["device"] = device
    
    return models

# --- TEXT PREPROCESSING ---

def clean_text(text):
    """Clean and preprocess the text"""
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text

def encode_text(text, word2idx, max_len=100):
    """Encode text for CNN model"""
    tokens = text.split()
    tokens = [word2idx.get(word, word2idx["<UNK>"]) for word in tokens]
    tokens = tokens[:max_len] + [word2idx["<PAD>"]] * max(0, max_len - len(tokens))
    return torch.tensor([tokens])

# --- PREDICTION FUNCTIONS ---

def predict_emotion(text, models):
    """Predict emotion from text using ensemble of models"""
    cleaned_text = clean_text(text)
    
    # CNN prediction
    encoded_text = encode_text(cleaned_text, models["word2idx"])
    encoded_text = encoded_text.to(models["device"])
    
    with torch.no_grad():
        cnn_output = models["cnn_model"](encoded_text)
        cnn_probs = torch.softmax(cnn_output, dim=1).cpu().numpy()
    
    # NB prediction
    nb_probs = models["nb_pipeline"].predict_proba([cleaned_text])
    
    # LightGBM prediction
    tfidf_vec = models["tfidf"].transform([cleaned_text])
    # Convert to DataFrame with proper feature names to avoid warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb_probs = models["lgb_model"].predict_proba(tfidf_vec)
    
    # Combine predictions
    X_blend = np.hstack([cnn_probs, nb_probs, lgb_probs])
    
    # Final prediction using blender
    final_probs = models["log_blender"].predict_proba(X_blend)[0]
    predicted_class = models["log_blender"].predict(X_blend)[0]
    emotion = models["label_encoder"].inverse_transform([predicted_class])[0]
    
    # Get probabilities for each emotion
    emotion_probs = dict(zip(models["label_encoder"].classes_, final_probs))
    
    return emotion, emotion_probs

# --- DATA MANAGEMENT FUNCTIONS ---

def load_dream_data():
    """Load dream data from file if it exists, otherwise create new DataFrame"""
    try:
        df = pd.read_csv("dream_journal.csv")
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle NaN values in string columns
        df['title'] = df['title'].fillna('')
        df['content'] = df['content'].fillna('')
        df['tags'] = df['tags'].fillna('')
        df['emotion'] = df['emotion'].fillna('neutral')
        df['emotion_probs'] = df['emotion_probs'].fillna('{}')
        
        # Handle boolean column
        if 'lucid' in df.columns:
            df['lucid'] = df['lucid'].fillna(False)
        else:
            df['lucid'] = False
            
        return df
    except FileNotFoundError:
        # Create a new DataFrame
        df = pd.DataFrame(columns=['date', 'title', 'content', 'emotion', 'emotion_probs', 'tags', 'lucid'])
        return df

def save_dream_data(df):
    """Save dream data to CSV file"""
    df.to_csv("dream_journal.csv", index=False)

# --- UI FUNCTIONS ---

def get_emotion_color(emotion):
    """Get color scheme based on predicted emotion"""
    color_map = {
        'happy': '#FFD700',     # Gold
        'sad': '#4682B4',       # Steel Blue
        'neutral': '#7CFC00',   # Lawn Green
        'nightmare': '#8B0000', # Dark Red
        'exciting': '#FF4500',  # Orange Red
        'peaceful': '#9370DB',  # Medium Purple
        'confusing': '#696969'  # Dim Gray
    }
    # Default to light gray if emotion not in map
    return color_map.get(emotion.lower(), '#D3D3D3')

def set_theme_based_on_emotion(emotion):
    """Set app theme based on predicted dream emotion"""
    color = get_emotion_color(emotion)
    
    # Apply custom theme
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {color}20;
    }}
    .st-emotion-cache-18ni7ap.ezrtsby2 {{
        background-color: {color}50;
    }}
    .stButton button {{
        background-color: {color};
        color: white;
    }}
    .stTextInput input, .stTextArea textarea {{
        border-color: {color};
    }}
    div[data-testid="stHeader"] {{
        background-color: {color}40;
    }}
    </style>
    """, unsafe_allow_html=True)

def display_audio_player(emotion):
    """Display audio player with music based on emotion"""
    # Define audio file based on emotion
    audio_map = {
        'happy': 'emotion_music/happy.mp3',
        'sad': 'emotion_music/sad.mp3',
        'neutral': 'emotion_music/neutral.mp3',
        'nightmare': 'emotion_music/nightmare.mp3',
        'exciting': 'emotion_music/exciting.mp3',
        'peaceful': 'emotion_music/peaceful.mp3',
        'confusing': 'emotion_music/confusing.mp3'
    }
    
    audio_file = audio_map.get(emotion.lower(), 'emotion_music/neutral.mp3')
    
    # Check if file exists
    if os.path.exists(audio_file):
        st.audio(audio_file, format='audio/mp3')
    else:
        st.info(f"Music for {emotion} emotion not found. Place audio files in the 'emotion_music' folder.")

# --- VISUALIZATION FUNCTIONS ---

def plot_emotion_timeline(df):
    """Plot emotion timeline of dreams"""
    if len(df) == 0:
        return st.info("Add dreams to see timeline analysis")
    
    # Group by date and emotion, count occurrences
    emotion_counts = df.groupby(['date', 'emotion']).size().reset_index(name='count')
    
    # Create plot
    fig = px.scatter(emotion_counts, x='date', y='emotion', size='count',
                     color='emotion', opacity=0.7,
                     title='Dream Emotions Timeline',
                     labels={'date': 'Date', 'emotion': 'Emotion', 'count': 'Count'},
                     height=400)
    
    # Add line connecting same emotions
    for emotion in df['emotion'].unique():
        emotion_data = emotion_counts[emotion_counts['emotion'] == emotion]
        if len(emotion_data) > 1:  # Only add line if more than one point
            fig.add_trace(go.Scatter(
                x=emotion_data['date'],
                y=emotion_data['emotion'],
                mode='lines',
                line=dict(width=1, dash='dot'),
                showlegend=False
            ))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Emotion')
    st.plotly_chart(fig, use_container_width=True)

def plot_emotion_distribution(df):
    """Plot distribution of emotions"""
    if len(df) == 0:
        return st.info("Add dreams to see emotion distribution")
    
    # Count emotions
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['emotion', 'count']
    
    # Create pie chart
    fig = px.pie(emotion_counts, values='count', names='emotion',
                 title='Dream Emotion Distribution',
                 color='emotion')
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def plot_emotion_confidence(df):
    """Plot confidence of emotion predictions over time"""
    if len(df) == 0 or 'emotion_probs' not in df.columns:
        return st.info("Add dreams to see confidence analysis")
    
    # Convert string representation of dict to actual dict and handle errors
    def safe_eval_probs(prob_str):
        try:
            if isinstance(prob_str, str) and prob_str.strip():
                return eval(prob_str)
            else:
                return {}
        except:
            return {}
    
    df['emotion_probs'] = df['emotion_probs'].apply(safe_eval_probs)
    
    # Extract confidence for predicted emotion
    df['confidence'] = df.apply(
        lambda row: row['emotion_probs'].get(row['emotion'], 0) 
        if isinstance(row['emotion_probs'], dict) else 0, 
        axis=1
    )
    
    # Create line chart
    fig = px.line(df.sort_values('date'), x='date', y='confidence',
                  color='emotion', markers=True,
                  title='Emotion Prediction Confidence Over Time',
                  labels={'date': 'Date', 'confidence': 'Prediction Confidence'})
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Confidence')
    st.plotly_chart(fig, use_container_width=True)

def plot_lucidity_trends(df):
    """Plot lucid dream trends over time"""
    if len(df) == 0 or 'lucid' not in df.columns:
        return st.info("Add dreams with lucidity information to see trends")
    
    # Group by date and calculate percentage of lucid dreams
    lucid_df = df.groupby(pd.Grouper(key='date', freq='ME')).agg(
        total_dreams=('lucid', 'count'),
        lucid_dreams=('lucid', lambda x: sum(x == True))
    ).reset_index()
    
    lucid_df['lucid_percentage'] = lucid_df['lucid_dreams'] / lucid_df['total_dreams'] * 100
    
    # Create bar chart
    fig = px.bar(lucid_df, x='date', y=['total_dreams', 'lucid_dreams'],
                 title='Lucid Dream Trends by Month',
                 labels={'date': 'Month', 'value': 'Number of Dreams', 'variable': 'Dream Type'},
                 color_discrete_map={'total_dreams': 'lightblue', 'lucid_dreams': 'darkblue'})
    
    # Add line for percentage
    fig.add_trace(go.Scatter(
        x=lucid_df['date'],
        y=lucid_df['lucid_percentage'],
        mode='lines+markers',
        name='Lucid %',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    # Add secondary y-axis
    fig.update_layout(
        yaxis2=dict(
            title='Lucid Dream %',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        xaxis_title='Month',
        yaxis_title='Number of Dreams'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---

def main():
    # Attempt to load models
    try:
        models = load_models()
        models_loaded = True
    except FileNotFoundError:
        models_loaded = False
        st.error("Model files not found. Please ensure all model files are in the app directory.")
    
    # Load dream data
    dream_data = load_dream_data()

    # Sidebar
    st.sidebar.title("🌙 Dream Journal & Analysis")
    
    # Navigation
    page = st.sidebar.radio("Navigate", ["Journal Entry", "Dream Archive", "Analysis Dashboard", "Settings"])
    
    # Get last emotion for theme if available
    if len(dream_data) > 0 and 'emotion' in dream_data.columns:
        last_emotion = dream_data.iloc[-1]['emotion'] if len(dream_data) > 0 else "neutral"
        set_theme_based_on_emotion(last_emotion)
    
    # Journal Entry Page
    if page == "Journal Entry":
        st.title("✏️ Record New Dream")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Form for dream entry
            with st.form("dream_entry_form"):
                dream_date = st.date_input("Dream Date", datetime.date.today())
                dream_title = st.text_input("Dream Title")
                dream_content = st.text_area("Dream Description", height=200, 
                                           placeholder="Describe your dream in detail...")
                
                col_tags, col_lucid = st.columns(2)
                with col_tags:
                    dream_tags = st.text_input("Tags (comma separated)")
                with col_lucid:
                    lucid_dream = st.checkbox("Lucid Dream")
                
                submit_button = st.form_submit_button("Analyze & Save Dream")
            
            # Process form submission
            if submit_button and dream_content:
                if models_loaded:
                    with st.spinner("Analyzing dream emotion..."):
                        # Predict emotion
                        emotion, emotion_probs = predict_emotion(dream_content, models)
                        
                        # Prepare data for saving
                        new_dream = {
                            'date': pd.Timestamp(dream_date),
                            'title': dream_title if dream_title else '',
                            'content': dream_content,
                            'emotion': emotion,
                            'emotion_probs': str(emotion_probs),  # Convert dict to string for storage
                            'tags': dream_tags if dream_tags else '',
                            'lucid': lucid_dream
                        }
                        
                        # Add to dataframe - handle empty dataframe properly
                        if len(dream_data) == 0:
                            dream_data = pd.DataFrame([new_dream])
                        else:
                            # Use pd.concat with proper handling for future compatibility
                            new_df = pd.DataFrame([new_dream])
                            dream_data = pd.concat([dream_data, new_df], ignore_index=True)
                        save_dream_data(dream_data)
                        
                        # Update theme based on predicted emotion
                        set_theme_based_on_emotion(emotion)
                        
                        # Success message
                        st.success(f"Dream saved and analyzed! Detected emotion: {emotion}")
                        
                        # Show emotion probabilities
                        st.write("### Emotion Probabilities")
                        prob_df = pd.DataFrame({
                            'Emotion': list(emotion_probs.keys()),
                            'Probability': list(emotion_probs.values())
                        }).sort_values('Probability', ascending=False)
                        
                        fig = px.bar(prob_df, x='Emotion', y='Probability', 
                                     color='Emotion', title='Emotion Analysis Results')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Play emotion-based music
                        st.write("### Dream Soundtrack")
                        st.write(f"Playing music based on detected emotion: {emotion}")
                        display_audio_player(emotion)
                else:
                    st.error("Cannot analyze dream. Models are not loaded correctly.")
        
        with col2:
            if len(dream_data) > 0:
                st.subheader("Recent Dreams")
                recent_dreams = dream_data.sort_values('date', ascending=False).head(5)
                for i, dream in recent_dreams.iterrows():
                    with st.container():
                        st.markdown(f"**{dream['title'] if dream['title'] else 'Untitled'}** - {dream['date'].strftime('%Y-%m-%d')}")
                        st.markdown(f"*{dream['emotion']}*")
                        st.markdown("---")
    
    # Dream Archive Page
    elif page == "Dream Archive":
        st.title("📚 Dream Archive")
        
        if len(dream_data) == 0:
            st.info("No dreams recorded yet. Go to Journal Entry to add your first dream!")
        else:
            # Search and filter
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("Search dreams", "")
            with col2:
                emotion_filter = st.multiselect("Filter by emotion", 
                                              options=sorted(dream_data['emotion'].unique()))
            with col3:
                date_range = st.date_input("Date range", 
                                         [dream_data['date'].min().date(), dream_data['date'].max().date()])
            
            # Apply filters
            filtered_data = dream_data.copy()
            
            # Search filter
            if search_term:
                filtered_data = filtered_data[
                    (filtered_data['content'].str.contains(search_term, case=False, na=False)) | 
                    (filtered_data['title'].str.contains(search_term, case=False, na=False))
                ]
            
            # Emotion filter
            if emotion_filter:
                filtered_data = filtered_data[filtered_data['emotion'].isin(emotion_filter)]
            
            # Date filter
            filtered_data = filtered_data[(filtered_data['date'].dt.date >= date_range[0]) & 
                                        (filtered_data['date'].dt.date <= date_range[1])]
            
            # Display results
            st.write(f"Found {len(filtered_data)} dreams")
            
            # Display dreams
            for i, dream in filtered_data.sort_values('date', ascending=False).iterrows():
                with st.expander(f"{dream['date'].strftime('%Y-%m-%d')} - {dream['title'] if dream['title'] else 'Untitled'} ({dream['emotion']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(dream['content'])
                        
                        # Safe handling of tags
                        if pd.notna(dream['tags']) and dream['tags'].strip():
                            st.write("**Tags:** " + str(dream['tags']))
                        
                        # Safe handling of lucid flag
                        if pd.notna(dream['lucid']) and dream['lucid']:
                            st.write("**Lucid Dream** ✨")
                    
                    with col2:
                        # Display emotion probabilities if available
                        if 'emotion_probs' in dream and pd.notna(dream['emotion_probs']) and str(dream['emotion_probs']).strip():
                            try:
                                probs = eval(str(dream['emotion_probs'])) if isinstance(dream['emotion_probs'], str) else dream['emotion_probs']
                                if probs and isinstance(probs, dict):
                                    probs_df = pd.DataFrame({
                                        'Emotion': list(probs.keys()),
                                        'Probability': list(probs.values())
                                    }).sort_values('Probability', ascending=False)
                                    
                                    fig = px.pie(probs_df, values='Probability', names='Emotion', 
                                                title='Emotion Breakdown', hole=0.3)
                                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                                    st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.write("Emotion probabilities not available")
                        
                        # Add edit and delete buttons
                        if st.button("Play Music", key=f"music_{i}"):
                            display_audio_player(dream['emotion'])
                        
                        if st.button("Delete", key=f"delete_{i}"):
                            # Remove from dataframe
                            dream_data = dream_data.drop(i)
                            save_dream_data(dream_data)
                            st.success("Dream deleted!")
                            st.rerun()
    
    # Analysis Dashboard Page
    elif page == "Analysis Dashboard":
        st.title("📊 Dream Analysis Dashboard")
        
        if len(dream_data) == 0:
            st.info("No dreams recorded yet. Add dreams to see analysis.")
        else:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Dreams", len(dream_data))
            with col2:
                st.metric("Unique Emotions", dream_data['emotion'].nunique())
            with col3:
                if 'lucid' in dream_data.columns:
                    lucid_count = dream_data['lucid'].sum()
                    lucid_pct = lucid_count / len(dream_data) * 100
                    st.metric("Lucid Dreams", f"{lucid_count} ({lucid_pct:.1f}%)")
            with col4:
                most_common = dream_data['emotion'].value_counts().index[0]
                st.metric("Most Common Emotion", most_common)
            
            # Tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Distribution", "Confidence", "Lucidity"])
            
            with tab1:
                plot_emotion_timeline(dream_data)
            
            with tab2:
                plot_emotion_distribution(dream_data)
            
            with tab3:
                plot_emotion_confidence(dream_data)
            
            with tab4:
                plot_lucidity_trends(dream_data)
            
            # Dream content analysis
            st.subheader("Dream Content Analysis")
            st.write("Word frequency in dream descriptions:")
            
            # Process text for word cloud
            if len(dream_data) > 0:
                all_text = " ".join(dream_data['content'].fillna('').tolist())
                all_text = clean_text(all_text)
                
                if all_text.strip():  # Only proceed if there's actual text
                    # Get word frequencies
                    words = all_text.split()
                    word_counts = pd.Series(words).value_counts().head(20)
                    
                    # Plot word frequencies
                    fig = px.bar(x=word_counts.index, y=word_counts.values,
                                labels={'x': 'Word', 'y': 'Frequency'},
                                title='Top 20 Words in Dreams')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No dream content available for analysis.")
    
    # Settings Page
    elif page == "Settings":
        st.title("⚙️ Settings")
        
        # Theme settings
        st.subheader("Theme Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            if len(dream_data) > 0 and 'emotion' in dream_data.columns:
                emotions = sorted(dream_data['emotion'].unique())
                selected_emotion = st.selectbox("Preview theme for emotion", emotions)
                set_theme_based_on_emotion(selected_emotion)
                
                # Test audio player
                st.write("Test emotion-based music:")
                display_audio_player(selected_emotion)
        
        with col2:
            st.write("Theme colors by emotion:")
            for emotion in ['happy', 'sad', 'neutral', 'nightmare', 'exciting', 'peaceful', 'confusing']:
                color = get_emotion_color(emotion)
                st.markdown(f"""
                <div style="background-color: {color}; 
                            color: white; 
                            padding: 10px; 
                            border-radius: 5px; 
                            margin-bottom: 5px;">
                    {emotion.capitalize()}
                </div>
                """, unsafe_allow_html=True)
        
        # Data management
        st.subheader("Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Dream Data"):
                # Create download link
                csv = dream_data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="dream_journal_export.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Also provide JSON export
                json_data = dream_data.to_json(orient='records', date_format='iso')
                b64_json = base64.b64encode(json_data.encode()).decode()
                href_json = f'<a href="data:file/json;base64,{b64_json}" download="dream_journal_export.json">Download JSON</a>'
                st.markdown(href_json, unsafe_allow_html=True)
        
        with col2:
            if st.button("Clear All Data", type="primary"):
                confirm = st.checkbox("I understand this will delete all dream entries")
                if confirm:
                    # Create empty dataframe
                    dream_data = pd.DataFrame(columns=['date', 'title', 'content', 'emotion', 'emotion_probs', 'tags', 'lucid'])
                    save_dream_data(dream_data)
                    st.success("All dream data cleared!")
                    st.rerun()

# Run the app
if __name__ == "__main__":
    main()