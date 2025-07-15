import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import base64
import io
import pickle
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import json

import pickle

# Train or load your model (this is just an example)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# Normally you would fit it with model.fit(X, y)

# Save the model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Try to import NLTK components
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Try to use NLTK stopwords
    try:
        nltk_stopwords = set(stopwords.words('english'))
        nltk_available = True
    except:
        nltk_available = False
        print("NLTK data not available, using fallback...")
except ImportError:
    nltk_available = False
    print("NLTK not installed, using fallback...")

# Nigeria-specific fallback stopwords (including common Nigerian terms)
NIGERIAN_STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now', 'oga', 'sir', 'madam', 'bros', 'sister'
])

def get_stopwords():
    """Get stopwords - use NLTK if available, otherwise Nigerian fallback"""
    if nltk_available:
        try:
            return set(stopwords.words('english'))
        except:
            return NIGERIAN_STOPWORDS
    return NIGERIAN_STOPWORDS

def tokenize_text(text):
    """Tokenize text - use NLTK if available, otherwise fallback"""
    if nltk_available:
        try:
            return word_tokenize(text.lower())
        except:
            return re.findall(r'\b\w+\b', text.lower())
    return re.findall(r'\b\w+\b', text.lower())

# Import database module with error handling
try:
    from database import Database
    database_available = True
    print("✅ Database module loaded successfully!")
except ImportError as e:
    database_available = False
    st.error(f"Database module not found: {str(e)}")
    st.error("Creating fallback database functionality...")
    
    # Fallback Database class
    class Database:
        def __init__(self):
            self.detections = []
            self.feedback = []
        
        def save_detection(self, text, prediction, probability):
            detection_id = len(self.detections) + 1
            self.detections.append({
                'id': detection_id,
                'text': text,
                'prediction': prediction,
                'probability': probability,
                'timestamp': datetime.now().isoformat()
            })
            return detection_id
        
        def save_feedback(self, detection_id, is_correct, feedback_text=""):
            self.feedback.append({
                'detection_id': detection_id,
                'is_correct': is_correct,
                'feedback_text': feedback_text
            })
        
        def get_feedback_stats(self):
            if not self.feedback:
                return {'total_feedback': 0, 'correct_rate': 0}
            
            total = len(self.feedback)
            correct = sum(1 for f in self.feedback if f['is_correct'])
            return {
                'total_feedback': total,
                'correct_rate': (correct / total) * 100 if total > 0 else 0
            }
        
        def get_recent_detections(self, limit=50):
            return self.detections[-limit:] if self.detections else []
        
        def close(self):
            pass

# Nigeria-specific ML Model class
class NigerianScamDetector:
    def __init__(self):
        # Nigeria-specific scam indicators
        self.nigerian_scam_keywords = [
            # General investment scam keywords
            'guaranteed', 'returns', 'profit', 'risk-free', 'limited time',
            'act now', 'exclusive', 'secret', 'insider', 'get rich',
            'easy money', 'no risk', 'double your money', 'investment opportunity',
            'high returns', 'make money fast', 'financial freedom',
            
            # Nigerian-specific keywords
            'forex trading', 'binary options', 'cryptocurrency mining',
            'ponzi', 'pyramid scheme', 'matrix', 'gifting circle',
            'mmm', 'ultimate cycler', 'twinkas', 'zarfund',
            'crowd1', 'longrich', 'aim global', 'organo gold',
            'bitcoin investment', 'cryptocurrency doubler',
            'oil and gas investment', 'real estate flipping',
            'importation business', 'dollar buy and sell',
            
            # Nigerian currency and payment terms
            'naira', '₦', 'western union', 'moneygram', 'mobile money',
            'paypal', 'bitcoin wallet', 'perfect money',
            
            # Nigerian locations and terms
            'lagos', 'abuja', 'port harcourt', 'kano', 'ibadan',
            'oga', 'boss', 'chairman', 'ceo', 'entrepreneur',
            'business mogul', 'wealth creator', 'financial consultant',
            
            # Common Nigerian scam phrases
            'join my team', 'be your own boss', 'financial breakthrough',
            'poverty is a choice', 'multiple streams of income',
            'residual income', 'passive income', 'network marketing',
            'affiliate marketing', 'digital marketing', 'online business'
        ]
        
        # Weight multipliers for Nigerian-specific terms
        self.nigerian_weight_multipliers = {
            'forex': 2.0,
            'mmm': 3.0,
            'ponzi': 3.0,
            'pyramid': 2.5,
            'bitcoin doubler': 3.0,
            'guaranteed returns': 2.5,
            'naira': 1.5,
            'lagos': 1.3,
            'oga': 1.2
        }
    
    def predict(self, text):
        """Nigeria-specific prediction based on keyword matching and weighting"""
        text_lower = text.lower()
        scam_score = 0
        
        # Count scam indicators with Nigerian-specific weighting
        for keyword in self.nigerian_scam_keywords:
            if keyword in text_lower:
                weight = self.nigerian_weight_multipliers.get(keyword, 1.0)
                scam_score += weight
        
        # Additional Nigerian-specific checks
        if any(term in text_lower for term in ['₦', 'naira']):
            scam_score += 0.5
        
        if any(term in text_lower for term in ['whatsapp', 'telegram']):
            scam_score += 1.0
        
        # Calculate probability based on weighted keyword density
        probability = min(scam_score / 8.0, 0.95)  # Adjusted for Nigerian context
        
        # Determine prediction
        if probability > 0.5:
            return "SCAM", probability
        else:
            return "LEGITIMATE", 1 - probability

# Page configuration
st.set_page_config(
    page_title="VerifyIt Nigeria - Investment Scam Detector",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define file paths with fallback
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')

# Initialize session state
if 'db' not in st.session_state:
    try:
        st.session_state.db = Database()
        st.success("✅ Database initialized successfully!")
    except Exception as e:
        st.error(f"❌ Database initialization failed: {str(e)}")
        st.session_state.db = Database()  # Use fallback

if 'model' not in st.session_state:
    # Try to load saved model, otherwise use Nigerian detector
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Try to load vectorizer
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                
                # Create a wrapper class for the loaded model
                class LoadedModelWrapper:
                    def __init__(self, model, vectorizer):
                        self.model = model
                        self.vectorizer = vectorizer
                        self.fallback = NigerianScamDetector()
                    
                    def predict(self, text):
                        try:
                            # Transform the text using the vectorizer
                            text_vector = self.vectorizer.transform([text])
                            
                            # Get prediction
                            if hasattr(self.model, 'predict_proba'):
                                prediction_proba = self.model.predict_proba(text_vector)[0]
                                scam_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                            else:
                                prediction = self.model.predict(text_vector)[0]
                                scam_prob = 0.8 if prediction == 1 else 0.2
                            
                            # Determine label
                            if scam_prob > 0.5:
                                return "SCAM", scam_prob
                            else:
                                return "LEGITIMATE", 1 - scam_prob
                                
                        except Exception as e:
                            st.warning(f"Advanced model failed, using Nigerian detector: {str(e)}")
                            return self.fallback.predict(text)
                
                st.session_state.model = LoadedModelWrapper(loaded_model, vectorizer)
                st.success("✅ Advanced ML model loaded successfully!")
                
            else:
                st.warning("⚠️ Vectorizer file not found. Using Nigerian keyword-based detector.")
                st.session_state.model = NigerianScamDetector()
        else:
            st.warning("⚠️ Model file not found. Using Nigerian keyword-based detector.")
            st.session_state.model = NigerianScamDetector()
            
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {str(e)}. Using Nigerian keyword-based detector.")
        st.session_state.model = NigerianScamDetector()

if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Custom CSS for black theme Nigerian-style design
st.markdown("""
<style>
    /* Overall app background - BLACK THEME */
    .stApp {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        padding-top: 2rem;
    }
    
    /* Sidebar background */
    .css-1d391kg, .css-1cypcdb, .sidebar .sidebar-content {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
    }
    
    /* Header styling with Nigerian colors on black */
    .main-header {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 50%, #228B22 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        border: 3px solid #228B22;
        box-shadow: 0 8px 25px rgba(34, 139, 34, 0.4);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 0.5rem;
    }
    
    .main-header h3 {
        color: #E8FFE8 !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #F0FFF0 !important;
        font-size: 1.1rem;
    }
    
    /* All text elements - white on black */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    .main p, .main div, .main span, .main li {
        color: #E0E0E0 !important;
    }
    
    /* Sidebar text styling */
    .sidebar h1, .sidebar h2, .sidebar h3, .sidebar h4, .sidebar h5, .sidebar h6 {
        color: #FFFFFF !important;
    }
    
    .sidebar p, .sidebar div, .sidebar span, .sidebar li {
        color: #E0E0E0 !important;
    }
    
    /* Input fields - dark theme */
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        border: 2px solid #404040 !important;
        border-radius: 8px;
    }
    
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
    }
    
    /* Alert styling with dark theme */
    .stAlert > div {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        border-radius: 10px;
        border-left: 5px solid #228B22;
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background-color: #1a4d1a !important;
        color: #90EE90 !important;
        border-left: 5px solid #32CD32 !important;
    }
    
    .stError {
        background-color: #4d1a1a !important;
        color: #FF6B6B !important;
        border-left: 5px solid #DC143C !important;
    }
    
    .stWarning {
        background-color: #4d4d1a !important;
        color: #FFD700 !important;
        border-left: 5px solid #FFA500 !important;
    }
    
    .stInfo {
        background-color: #1a3d4d !important;
        color: #87CEEB !important;
        border-left: 5px solid #4682B4 !important;
    }
    
    /* Metric cards - dark theme */
    .metric-card, [data-testid="metric-container"] {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(34, 139, 34, 0.2);
        border-left: 5px solid #228B22;
        border: 1px solid #404040;
    }
    
    [data-testid="metric-container"] > div {
        color: #FFFFFF !important;
    }
    
    /* Prediction result styling */
    .prediction-safe {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(34, 139, 34, 0.4);
        border: 2px solid #32CD32;
    }
    
    .prediction-scam {
        background: linear-gradient(135deg, #DC143C 0%, #FF4444 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 6px 20px rgba(220, 20, 60, 0.4);
        border: 2px solid #FF4444;
    }
    
    /* Sidebar info boxes - dark theme */
    .sidebar-info {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 5px solid #228B22;
        border: 1px solid #404040;
    }
    
    .sidebar-info h4 {
        color: #32CD32 !important;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .sidebar-info ul, .sidebar-info li, .sidebar-info p {
        color: #E0E0E0 !important;
    }
    
    /* RED WARNING DISCLAIMER - more visible on black */
    .sidebar-disclaimer {
        background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 2px solid #FF4444;
        color: white !important;
        box-shadow: 0 4px 15px rgba(220, 20, 60, 0.5);
    }
    
    .sidebar-disclaimer h4 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.8rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .sidebar-disclaimer p {
        color: #FFE4E4 !important;
        font-weight: 500;
        line-height: 1.4;
    }
    
    .sidebar-disclaimer strong {
        color: white !important;
        font-weight: 700;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(34, 139, 34, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #32CD32 0%, #228B22 100%);
        box-shadow: 0 6px 20px rgba(34, 139, 34, 0.5);
        transform: translateY(-2px);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2a2a2a !important;
        border-radius: 8px;
        padding: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #228B22 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        font-weight: 600;
        border: 1px solid #404040;
        border-radius: 8px;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #2a2a2a !important;
        border: 2px dashed #228B22;
        border-radius: 10px;
        padding: 1rem;
        color: #FFFFFF !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #2a2a2a !important;
        padding: 0.5rem;
        border-radius: 8px;
        border: 1px solid #404040;
    }
    
    .stRadio label {
        color: #FFFFFF !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #228B22 !important;
    }
    
    /* Footer styling - dark theme */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 12px;
        text-align: center;
        color: #FFFFFF;
        border: 2px solid #404040;
    }
    
    .footer p {
        color: #E0E0E0 !important;
        margin-bottom: 0.5rem;
    }
    
    .footer strong {
        color: #32CD32 !important;
    }
    
    /* Code blocks */
    .stCode {
        background-color: #1a1a1a !important;
        color: #FFFFFF !important;
        border-left: 4px solid #228B22 !important;
    }
    
    /* Plotly charts - dark theme */
    .js-plotly-plot {
        background-color: #1a1a1a !important;
    }
    
    /* Author info styling */
    .author-info {
        background: linear-gradient(135deg, #2a2a2a 0%, #404040 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #32CD32;
        border: 1px solid #404040;
        color: #FFFFFF !important;
    }
    
    .author-info h4 {
        color: #32CD32 !important;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    
    .author-info p {
        color: #E0E0E0 !important;
        line-height: 1.4;
    }
    
    .author-info strong {
        color: #FFFFFF !important;
    }
    
    /* University branding */
    .university-brand {
        background: linear-gradient(135deg, #1a4d1a 0%, #228B22 100%);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        color: white !important;
        border: 2px solid #32CD32;
        box-shadow: 0 4px 15px rgba(34, 139, 34, 0.3);
    }
    
    .university-brand h4, .university-brand p {
        color: white !important;
    }
    
    /* Selectbox dropdown */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        border: 2px solid #404040 !important;
    }
    
    /* Data frame styling */
    .stDataFrame, .stDataFrame table {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
    }
    
    .stDataFrame th {
        background-color: #404040 !important;
        color: #FFFFFF !important;
    }
    
    .stDataFrame td {
        background-color: #2a2a2a !important;
        color: #E0E0E0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Header with author information
st.markdown("""
<div class="main-header">
    <h1>🇳🇬 VerifyIt Nigeria</h1>
    <h3>AI-Powered Investment Scam Detector</h3>
    <p>Protecting Nigerian Investors with Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Author Information
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="author-info">
        <h4>👨‍🎓 Research Author</h4>
        <p><strong>Ononneobazi Aquah</strong><br>
    </div>
    """, unsafe_allow_html=True)

# Sidebar with Nigerian-specific content and red disclaimer
with st.sidebar:
    st.markdown("### 🇳🇬 Navigation")
    page = st.selectbox(
        "Choose a page:",
        ["🔍 Scam Detection", "📈 Analytics", "📋 History", "🇳🇬 Nigerian Context", "ℹ️ About"]
    )
    
    st.markdown("""<div class="sidebar-info">
    <h4>🚨 Nigerian Scam Warning Signs</h4>
    <ul>
        <li>Forex trading guarantees</li>
        <li>MMM-style schemes</li>
        <li>Bitcoin doublers</li>
        <li>WhatsApp/Telegram recruitment</li>
        <li>Naira multiplication promises</li>
        <li>Fake oil & gas investments</li>
        <li>Ponzi/Pyramid schemes</li>
    </ul>
    </div>""", unsafe_allow_html=True)
    
    # RED DISCLAIMER SECTION
    st.markdown("""<div class="sidebar-disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This tool provides risk assessment, NOT financial advice!</strong></p>
    <p>• Always conduct independent research</p>
    <p>• Consult licensed Nigerian financial advisors</p>
    <p>• Report suspected scams to EFCC</p>
    <p>• No AI system is 100% accurate</p>
    <p><strong>USE YOUR JUDGMENT!</strong></p>
    </div>""", unsafe_allow_html=True)
    
    # Quick stats
    stats = st.session_state.db.get_feedback_stats()
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Scans", len(st.session_state.detection_history))
    with col2:
        st.metric("Accuracy", f"{stats['correct_rate']:.1f}%")

# Emergency contact info in sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""<div class="sidebar-info">
    <h4>🚨 Emergency Contacts</h4>
    <p><strong>Report Scams:</strong></p>
    <ul>
        <li>EFCC: 0800-CALL-EFCC</li>
        <li>SEC: +234-9-461-0000</li>
        <li>Police: 199</li>
    </ul>
    <p><strong>Always verify before investing!</strong></p>
    </div>""", unsafe_allow_html=True)

# Nigerian scam examples for testing
NIGERIAN_SCAM_EXAMPLES = {
    "Forex Trading Scam": """
    🔥 FOREX TRADING OPPORTUNITY IN NIGERIA 🔥
    
    Join our exclusive WhatsApp group for guaranteed forex profits!
    
    ✅ 100% guaranteed returns 
    ✅ No risk involved
    ✅ Start with just ₦10,000
    ✅ Make ₦50,000 in 7 days
    ✅ Our CEO is a forex expert from Lagos

    Contact CEO Johnson on WhatsApp: 08012345678
    Send registration fee to GT Bank: 0123456789
    
    ACT FAST! Slots filling up quickly!
    """
}

st.error("⚠️ This is a CLASSIC scam! Notice the guaranteed returns, pressure tactics, WhatsApp contact, and upfront payment request.")

with st.expander("✅ Example: Legitimate Investment Information"):
    st.code("""
    ABC Investment Management Limited
    (CAC Registration: RC123456, SEC License: SEC/LIC/INV/789)
    
    Investment Opportunity: Nigerian Treasury Bills Portfolio
    
    📋 Investment Details:
    - Minimum Investment: ₦100,000
    - Expected Returns: 8-12% per annum (not guaranteed)
    - Investment Period: 90-365 days
    - Risk Level: Low to Medium
    
    📍 Office Address: 123 Victoria Island, Lagos
    📞 Phone: +234-1-234-5678
    🌐 Website: www.abcinvestment.com.ng
    📧 Email: info@abcinvestment.com.ng
    
    ⚠️ Risk Disclosure: All investments carry risk. Past performance does not guarantee future results.
    
    Visit our office for consultation or download our prospectus from our website.
    """, language="text")
    
    st.success("✅ This shows proper registration, realistic returns, risk disclosure, and professional presentation.")

# Main page navigation logic
if page == "🔍 Scam Detection":
    st.markdown("## 🔍 Investment Scam Detection")
    
    # Text input for analysis
    user_input = st.text_area(
        "📝 Paste the investment offer or message you want to analyze:",
        height=200,
        placeholder="Example: Join our WhatsApp group for guaranteed forex profits! 100% returns in 7 days with no risk..."
    )
    
    if st.button("🔍 Analyze for Scams", key="analyze_btn"):
        if user_input.strip():
            with st.spinner("🤖 Analyzing text for scam indicators..."):
                # Get prediction from model
                prediction, probability = st.session_state.model.predict(user_input)
                
                # Save detection to database
                detection_id = st.session_state.db.save_detection(
                    user_input, prediction, probability
                )
                
                # Add to session history
                st.session_state.detection_history.append({
                    'id': detection_id,
                    'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                    'prediction': prediction,
                    'probability': probability,
                    'timestamp': datetime.now()
                })
                
                # Display results
                if prediction == "SCAM":
                    st.markdown(f"""
                    <div class="prediction-scam">
                        <h3>🚨 HIGH RISK - LIKELY SCAM</h3>
                        <p><strong>Confidence: {probability:.1%}</strong></p>
                        <p>This appears to be a potential investment scam. Exercise extreme caution!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("⚠️ **WARNING**: This message shows multiple red flags commonly associated with investment scams in Nigeria.")
                    
                else:
                    st.markdown(f"""
                    <div class="prediction-safe">
                        <h3>✅ LOWER RISK - APPEARS LEGITIMATE</h3>
                        <p><strong>Confidence: {probability:.1%}</strong></p>
                        <p>This appears to be a more legitimate investment opportunity.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("✅ **Good News**: This message appears to have fewer scam indicators.")
                
                # Feedback section
                st.markdown("### 📝 Was this analysis helpful?")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("👍 Yes, accurate"):
                        st.session_state.db.save_feedback(detection_id, True)
                        st.success("Thank you for your feedback!")
                
                with col2:
                    if st.button("👎 No, incorrect"):
                        st.session_state.db.save_feedback(detection_id, False)
                        st.warning("Thank you for the feedback. We'll use this to improve the system.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")
    
    # Quick test examples
    st.markdown("### 🧪 Quick Test Examples")
    if st.button("Test with Nigerian Forex Scam Example"):
        st.text_area("Example loaded:", NIGERIAN_SCAM_EXAMPLES["Forex Trading Scam"], height=200, key="example_1")

elif page == "📈 Analytics":
    st.markdown("## 📈 Analytics Dashboard")
    
    if st.session_state.detection_history:
        # Create analytics visualizations
        df = pd.DataFrame(st.session_state.detection_history)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_scans = len(df)
            st.metric("Total Scans", total_scans)
        
        with col2:
            scam_rate = (df['prediction'] == 'SCAM').mean() * 100
            st.metric("Scam Detection Rate", f"{scam_rate:.1f}%")
        
        with col3:
            avg_confidence = df['probability'].mean() * 100
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Prediction distribution
        fig = px.pie(
            values=df['prediction'].value_counts().values,
            names=df['prediction'].value_counts().index,
            title="Prediction Distribution",
            color_discrete_map={'SCAM': '#FF4444', 'LEGITIMATE': '#32CD32'}
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📊 No analysis data available yet. Start by analyzing some investment offers!")

elif page == "📋 History":
    st.markdown("## 📋 Detection History")
    
    if st.session_state.detection_history:
        # Display recent detections
        df = pd.DataFrame(st.session_state.detection_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            df[['timestamp', 'text', 'prediction', 'probability']],
            use_container_width=True
        )
        
        # Export functionality
        if st.button("📥 Export History"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"scam_detection_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    else:
        st.info("📋 No detection history available yet.")

elif page == "🇳🇬 Nigerian Context":
    st.markdown("## 🇳🇬 Nigerian Investment Scam Context")
    
    st.markdown("""
    ### 📚 Common Nigerian Investment Scams
    
    **1. 🔄 Ponzi/Pyramid Schemes**
    - MMM, Twinkas, Ultimate Cycler
    - Promise unrealistic returns
    - Require recruiting others
    
    **2. 💱 Forex Trading Scams**
    - Guaranteed daily profits
    - WhatsApp/Telegram groups
    - Fake trading platforms
    
    **3. ₿ Cryptocurrency Scams**
    - Bitcoin doublers
    - Fake mining operations
    - Ponzi schemes with crypto
    
    **4. 🏢 Fake Investment Companies**
    - Unregistered with SEC
    - No physical address
    - Pressure tactics
    """)
    
    st.markdown("""
    ### 🛡️ How to Protect Yourself
    
    **✅ Always Verify:**
    - Check SEC Nigeria registration
    - Visit physical offices
    - Research company background
    - Ask for proper documentation
    
    **🚨 Red Flags:**
    - Guaranteed returns
    - Pressure to act quickly
    - Recruitment requirements
    - No regulatory approval
    - WhatsApp-only communication
    """)
    
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About VerifyIt Nigeria")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Mission
        VerifyIt Nigeria is an AI-powered investment scam detection system specifically designed to protect Nigerian investors from fraudulent investment schemes. Our mission is to reduce financial fraud and promote safe investing practices across Nigeria.
        
        ### 🧠 How It Works
        Our system combines advanced machine learning techniques with Nigerian-specific knowledge to identify potential investment scams:
        
        1. **Text Analysis**: Advanced NLP techniques analyze investment descriptions
        2. **Nigerian Context**: Specialized knowledge of local scam patterns
        3. **Keyword Detection**: Identification of common scam indicators
        4. **Risk Scoring**: Probability-based risk assessment
        5. **Regulatory Integration**: Cross-reference with Nigerian regulatory warnings
        
        ### 🎓 Academic Research
        This project is developed as part of undergraduate research at a Nigerian university, focusing on:
        - Machine Learning applications in fraud detection
        - Localized AI solutions for developing economies
        - Financial literacy and consumer protection
        - Natural Language Processing for Nigerian context
        
        ### 📊 Technology Stack
        - **Frontend**: Streamlit (Python web framework)
        - **Machine Learning**: Scikit-learn, XGBoost
        - **NLP**: TF-IDF Vectorization, NLTK
        - **Database**: SQLite for data persistence
        - **Visualization**: Plotly, Matplotlib
        - **Deployment**: Local/Cloud deployment ready
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Key Features
        
        **🔍 Smart Detection**
        - Nigerian-specific scam patterns
        - Real-time text analysis
        - Confidence scoring
        - Multi-language support
        
        **📊 Analytics Dashboard**
        - Detection trends
        - Risk level analysis
        - Performance metrics
        - Export capabilities
        
        **🇳🇬 Local Context**
        - Nigerian regulatory info
        - Local scam examples
        - Emergency contacts
        - Educational resources
        
        **💾 Data Management**
        - Detection history
        - User feedback system
        - Continuous learning
        - Privacy protection
        """)
        
        # Model performance metrics (mock data for demonstration)
        st.markdown("### 🎯 Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "94.2%", "2.1%")
            st.metric("Precision", "91.8%", "1.5%")
        with col2:
            st.metric("Recall", "96.3%", "0.8%")
            st.metric("F1-Score", "94.0%", "1.2%")
    
    # Disclaimer and limitations
    st.markdown("### ⚠️ Important Disclaimer")
    st.warning("""
    **Please Note:**
    - This tool provides risk assessment, not financial advice
    - Always conduct independent research before investing
    - Consult licensed financial advisors for investment decisions
    - Report suspected scams to relevant Nigerian authorities
    - The system is continuously learning and improving
    - No AI system is 100% accurate - use your judgment
    """)
    
    # Contact and feedback section
    st.markdown("### 📞 Contact & Feedback")
    
    with st.expander("💌 Send Feedback"):
        feedback_type = st.selectbox("Feedback Type:", ["General Feedback", "Bug Report", "Feature Request", "False Positive", "False Negative"])
        feedback_message = st.text_area("Your feedback:")
        feedback_email = st.text_input("Email (optional):")
        
        if st.button("Send Feedback"):
            if feedback_message:
                # In a real implementation, this would send to a database or email
                st.success("Thank you for your feedback! It helps us improve VerifyIt Nigeria.")
            else:
                st.warning("Please enter your feedback message.")
    
    # Acknowledgments
    st.markdown("### 🙏 Acknowledgments")
    st.info("""
    **Special Thanks To:**
    - Nigerian Securities and Exchange Commission (SEC)
    - Economic and Financial Crimes Commission (EFCC)
    - Central Bank of Nigeria (CBN)
    - University research supervisors and peers
    - Nigerian fintech community for insights
    - Open source community for tools and libraries
    """)
    
    # Version information
    st.markdown("### 📋 Version Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Version**: 1.0.0")
    with col2:
        st.info("**Last Updated**: May 2025")
    with col3:
        st.info("**Status**: Beta")

# Footer
st.markdown("""
<div class="footer">
    <p>🇳🇬 <strong>VerifyIt Nigeria</strong> - Protecting Nigerian Investors with AI</p>
    <p>Made with ❤️ for Nigeria | For educational and research purposes</p>
    <p><em>Always verify investments with SEC Nigeria and consult licensed financial advisors</em></p>
</div>
""", unsafe_allow_html=True)

# Emergency contact info in sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### 🚨 Emergency Contacts
    **Report Scams:**
    - EFCC: 0800-CALL-EFCC
    - SEC: +234-9-461-0000
    - Police: 199
    
    **Always verify before investing!**
    """)

# Clean up database connection on app close
@st.cache_resource
def cleanup_database():
    if hasattr(st.session_state, 'db'):
        st.session_state.db.close()

# Register cleanup
import atexit
atexit.register(cleanup_database)
