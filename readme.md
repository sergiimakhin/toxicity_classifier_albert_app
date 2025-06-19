# 🛡️ Toxic Comment Classifier

**🌐 Live App**: https://toxic-classifier-app.streamlit.app/

An AI-powered web application for detecting toxic content in text using an ensemble of ALBERT transformer models. Built with Streamlit and PyTorch, this tool provides real-time analysis across multiple toxicity categories with confidence scoring and visual analytics.

## 🌟 Features

- **Multi-Category Detection**: Analyzes text across 6 distinct toxicity categories
- **Ensemble Learning**: Uses 3 cross-validated ALBERT models for robust predictions
- **Interactive Dashboard**: Real-time analysis with visual charts and confidence scores
- **Customizable Thresholds**: Adjustable sensitivity settings for different use cases
- **Example Templates**: Pre-built samples for quick testing
- **Theme-Adaptive UI**: Responsive design that adapts to light/dark themes
- **Uncertainty Quantification**: Shows prediction confidence intervals

## 🎯 Detection Categories

| Category | Description |
|----------|-------------|
| **General Toxicity** | Contains generally toxic or harmful language |
| **Severe Toxicity** | Contains extremely toxic or hateful content |
| **Obscene Language** | Contains profanity or sexually explicit content |
| **Threatening Language** | Contains threatening or intimidating language |
| **Insulting Content** | Contains insulting or demeaning language |
| **Identity-based Hate** | Contains hate speech targeting identity groups |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Streamlit

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd toxic-comment-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up model files**
   ```
   outputs/
   ├── tokenizer/
   │   ├── config.json
   │   ├── tokenizer.json
   │   └── tokenizer_config.json
   ├── model_fold0.pt
   ├── model_fold1.pt
   └── model_fold2.pt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📦 Dependencies

```python
streamlit
torch
transformers
numpy
matplotlib
plotly
```

## 🏗️ Architecture

### Model Architecture
- **Base Model**: ALBERT Base v2
- **Ensemble Size**: 3 models (cross-validation folds)
- **Max Sequence Length**: 128 tokens
- **Output**: Multi-label classification (6 categories)

### Technical Stack
- **Frontend**: Streamlit with custom CSS
- **Backend**: PyTorch + Transformers
- **Visualization**: Plotly for interactive charts
- **Device Support**: CPU/CUDA adaptive

## 💻 Usage

### Basic Analysis
1. Enter or paste text in the input area
2. Click "🔍 Analyze Text"
3. View results in the interactive dashboard

### Advanced Settings
- **Threshold Adjustment**: Use the sidebar slider to modify classification sensitivity
- **Raw Scores**: Enable to view detailed probability scores and uncertainty metrics
- **Quick Examples**: Test with pre-built samples representing different toxicity types

### API Integration
The prediction function can be used independently:

```python
results = predict_toxicity(text, models, tokenizer)
predictions = results['predictions']
uncertainties = results['uncertainty']
```

## 📊 Model Performance

The ensemble approach provides:
- **Improved Accuracy**: Multiple models reduce overfitting
- **Uncertainty Quantification**: Standard deviation across predictions
- **Robust Predictions**: Cross-validation ensures generalization

## 🔧 Configuration

Key configuration variables in the code:

```python
N_FOLDS = 3                    # Number of ensemble models
MAX_LEN = 128                  # Maximum sequence length
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 📁 Project Structure

```
toxic-comment-classifier/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── outputs/
    ├── tokenizer/            # ALBERT tokenizer files
    ├── model_fold0.pt        # Trained model (fold 0)
    ├── model_fold1.pt        # Trained model (fold 1)
    └── model_fold2.pt        # Trained model (fold 2)
```

## 🚀 Deployment

### Streamlit Cloud
1. Push your repository to GitHub
2. Ensure model files are under 100MB (use Git LFS if needed)
3. Connect your repo to Streamlit Cloud
4. Deploy with automatic dependency installation

### Local Deployment
```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## ⚡ Performance Optimization

- **Model Loading**: Cached with `@st.cache_resource`
- **Predictions**: Cached with `@st.cache_data`
- **Mixed Precision**: Automatic CUDA optimization when available
- **Efficient Tokenization**: Batch processing with attention masks

## 🎨 UI Features

- **Responsive Design**: Adapts to different screen sizes
- **Theme Support**: Compatible with Streamlit's light/dark themes
- **Interactive Charts**: Hover effects and threshold visualization
- **Color-Coded Results**: Intuitive red/green status indicators
- **Progress Indicators**: Loading states for better UX

## 📝 Example Use Cases

- **Content Moderation**: Automated screening of user comments
- **Social Media Monitoring**: Real-time toxicity detection
- **Educational Tools**: Teaching about online safety
- **Research Applications**: Analyzing discourse patterns
- **API Integration**: Embedding in larger applications

## ⚠️ Important Notes

- **Human Review Recommended**: This tool assists but doesn't replace human judgment
- **Bias Considerations**: Models may reflect training data biases
- **Context Limitations**: May not capture nuanced context or sarcasm
- **Language Support**: Optimized for English text
- **File Size Limits**: Model files must comply with deployment platform limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face Transformers**: For the ALBERT implementation
- **Streamlit**: For the web application framework
- **PyTorch**: For the deep learning backend
- **Plotly**: For interactive visualizations

## 📞 Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review existing discussions

---

**Disclaimer**: This tool is designed to assist in content moderation and should be used responsibly. Results should be reviewed by humans for critical decisions, especially in sensitive contexts.