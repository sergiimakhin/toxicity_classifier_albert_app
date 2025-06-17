import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from transformers import AutoTokenizer, AlbertForSequenceClassification
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Theme-adaptive styles */
    :root {
        --success-color: #28a745;
        --danger-color: #dc3545;
        --text-color: inherit;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border: 1px solid #f0ad4e;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #f0ad4e;
        color: #856404;
    }
    
    .safe-card {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #28a745;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        font-size: 16px;
        transition: border-color 0.3s;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
    }
    
    .prediction-section {
        background: rgba(248, 249, 250, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 2rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    
    /* Dark theme adaptations */
    @media (prefers-color-scheme: dark) {
        .prediction-section {
            background: rgba(40, 44, 52, 0.8);
            border: 1px solid rgba(255,255,255,0.1);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration ---
N_FOLDS = 3
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
LABEL_DISPLAY_NAMES = {
    'toxic': 'General Toxicity',
    'severe_toxic': 'Severe Toxicity',
    'obscene': 'Obscene Language',
    'threat': 'Threatening Language',
    'insult': 'Insulting Content',
    'identity_hate': 'Identity-based Hate'
}
LABEL_DESCRIPTIONS = {
    'toxic': 'Contains generally toxic or harmful language',
    'severe_toxic': 'Contains extremely toxic or hateful content',
    'obscene': 'Contains profanity or sexually explicit content',
    'threat': 'Contains threatening or intimidating language',
    'insult': 'Contains insulting or demeaning language',
    'identity_hate': 'Contains hate speech targeting identity groups'
}
MAX_LEN = 128
# Force CPU for Streamlit Cloud deployment, use CUDA locally if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_CUDA_AVAILABLE = torch.cuda.is_available()
TOKENIZER_PATH = "outputs/tokenizer"
MODEL_BASE_PATH = "outputs/model_fold"

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI-Powered Toxic Comment Detection</h1>
    <p>Advanced ensemble model using ALBERT transformers with cross-validation</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Architecture:** ALBERT Base v2  
    **Ensemble Size:** {N_FOLDS} models  
    **Device:** {DEVICE.type.upper()}  
    **Max Length:** {MAX_LEN} tokens  
    **Labels:** {len(LABELS)} categories
    """)
    
    st.markdown("### 🎯 Detection Categories")
    for label, desc in LABEL_DESCRIPTIONS.items():
        with st.expander(LABEL_DISPLAY_NAMES[label]):
            st.write(desc)
    
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 0.05)
    show_raw_scores = st.checkbox("Show Raw Probability Scores", value=False)

# --- Load Resources ---
@st.cache_resource
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {e}")
        return None

@st.cache_resource
def load_models():
    models = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for fold in range(N_FOLDS):
            status_text.text(f"Loading model {fold + 1}/{N_FOLDS}...")
            model_path = f"{MODEL_BASE_PATH}{fold}.pt"
            
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                st.error("Please ensure your model files are uploaded to the repository.")
                return None
                
            model = AlbertForSequenceClassification.from_pretrained(
                "albert-base-v2", 
                num_labels=len(LABELS)
            )
            
            # Load model weights with appropriate map_location
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()
            models.append(model)
            progress_bar.progress((fold + 1) / N_FOLDS)
        
        status_text.text("✅ All models loaded successfully!")
        return models
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.error("This might be due to missing model files or incompatible PyTorch versions.")
        return None

# Initialize models
with st.spinner("🔄 Loading AI models..."):
    tokenizer = load_tokenizer()
    models = load_models()

if tokenizer is None or models is None:
    st.error("❌ Failed to initialize the application. Please check your model files.")
    st.info("""
    **For Streamlit Cloud deployment:**
    1. Make sure your model files (`outputs/model_fold0.pt`, `outputs/model_fold1.pt`, `outputs/model_fold2.pt`) are in your repository
    2. Make sure your tokenizer files are in the `outputs/tokenizer/` directory
    3. Ensure all files are under GitHub's file size limits (100MB per file)
    4. Consider using Git LFS for large model files
    """)
    st.stop()

# --- Prediction Function ---
@st.cache_data
def predict_toxicity(text, _models, _tokenizer):
    encoded = _tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    fold_probs = []
    with torch.no_grad():
        # Only use autocast if CUDA is available
        if IS_CUDA_AVAILABLE:
            with torch.cuda.amp.autocast():
                for model in _models:
                    outputs = model(input_ids, attention_mask=attention_mask)
                    probs = torch.sigmoid(outputs.logits).cpu().numpy()
                    fold_probs.append(probs)
        else:
            for model in _models:
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                fold_probs.append(probs)

    avg_probs = np.mean(fold_probs, axis=0).flatten()
    std_probs = np.std(fold_probs, axis=0).flatten()
    
    return {
        'predictions': {label: float(prob) for label, prob in zip(LABELS, avg_probs)},
        'uncertainty': {label: float(std) for label, std in zip(LABELS, std_probs)}
    }

# --- Example Text Handling ---
example_texts = [
    "You're such an idiot, you don't know what you're talking about!",
    "If you say something like that one more time I am going to find you and make you suffer!",
    "All people from that country are disgusting and should go back where they came from.",
    "This is a thoughtful post, thanks for sharing your perspective.",
    "Your argument is flawed but I appreciate the discussion."
]

# Initialize example_selected
example_selected = None

# Handle example selection in sidebar first
with st.sidebar:
    st.markdown("### 📋 Quick Examples")
    for i, example in enumerate(example_texts):
        # Create descriptive labels for different toxicity types
        labels = ["🎯 Insult", "⚠️ Threat", "💢 Hate Speech", "✅ Positive", "🤝 Respectful"]
        if st.button(f"{labels[i]}", key=f"example_{i}", use_container_width=True):
            example_selected = example
            st.session_state.text_input = example

# --- Main Interface ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Text for Analysis")
    
    # Determine the value for the text area
    text_area_value = ""
    if example_selected is not None:
        text_area_value = example_selected
    else:
        text_area_value = st.session_state.get('text_input', '')
    
    user_input = st.text_area(
        "Type or paste the comment you want to analyze:",
        height=150,
        placeholder="Enter a comment here...",
        help="The model will analyze this text for various types of toxic content.",
        value=text_area_value
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        analyze_btn = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        if 'text_input' in st.session_state:
            del st.session_state.text_input
        st.rerun()



# Handle example selection from session state
if 'example_text' in st.session_state:
    user_input = st.session_state.example_text
    del st.session_state.example_text

# --- Analysis Results ---
if analyze_btn and user_input.strip():
    with st.spinner("🧠 Analyzing content..."):
        results = predict_toxicity(user_input, models, tokenizer)
        predictions = results['predictions']
        uncertainties = results['uncertainty']
    
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    
    # Overall Assessment
    max_prob = max(predictions.values())
    is_toxic_overall = max_prob >= threshold
    
    if is_toxic_overall:
        st.markdown(f"""
        <div class="warning-card">
            <h3>⚠️ Potentially Toxic Content Detected</h3>
            <p>Highest confidence: <strong>{max_prob:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-card">
            <h3>✅ Content Appears Safe</h3>
            <p>Highest toxicity score: <strong>{max_prob:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Results
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Detailed Analysis")
        
        # Create interactive chart
        labels = [LABEL_DISPLAY_NAMES[label] for label in LABELS]
        values = [predictions[label] * 100 for label in LABELS]
        
        # Create color scheme with good contrast for both themes
        colors = ['#28a745' if v < threshold*100 else '#dc3545' for v in values]
        text_colors = ['white' if v > 15 else 'black' for v in values]  # Dark text for low values
        
        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.2)', width=1)
            ),
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            textfont=dict(size=11, family='Arial'),
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Toxicity Classification Results",
            xaxis_title="Probability (%)",
            height=320,
            margin=dict(l=150, r=50, t=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='var(--text-color)')
        )
        
        fig.add_vline(x=threshold*100, line_dash="dash", line_color="rgba(102,102,102,0.8)", line_width=2,
                     annotation_text=f"Threshold ({threshold:.0%})", 
                     annotation=dict(font=dict(color='var(--text-color)')))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Results")
        
        for label in LABELS:
            prob = predictions[label]
            is_flagged = prob >= threshold
            
            # Adaptive display with theme-aware colors
            icon = "🚨" if is_flagged else "✅"
            status_color = "#dc3545" if is_flagged else "#28a745"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; 
                       padding: 6px 10px; margin: 3px 0; 
                       border: 1px solid {status_color}30; 
                       border-left: 4px solid {status_color}; 
                       background: {status_color}08; 
                       border-radius: 6px; font-size: 13px;">
                <span style="font-weight: 500;">{LABEL_DISPLAY_NAMES[label][:11]}{'...' if len(LABEL_DISPLAY_NAMES[label]) > 11 else ''}</span>
                <span style="color: {status_color}; font-weight: 600;">{icon} {prob:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Raw scores (if enabled)
    if show_raw_scores:
        st.markdown("### 🔢 Raw Probability Scores")
        score_data = []
        for label in LABELS:
            score_data.append({
                'Category': LABEL_DISPLAY_NAMES[label],
                'Probability': f"{predictions[label]:.4f}",
                'Percentage': f"{predictions[label]*100:.2f}%",
                'Uncertainty': f"±{uncertainties[label]:.4f}"
            })
        st.dataframe(score_data, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Timestamp
    st.markdown(f"*Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

elif analyze_btn and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>🛡️ <strong>Toxic Comment Classifier</strong> | Powered by ALBERT & Ensemble Learning</p>
    <p><small>This tool is designed to assist in content moderation. Human review is recommended for critical decisions.</small></p>
</div>
""", unsafe_allow_html=True)