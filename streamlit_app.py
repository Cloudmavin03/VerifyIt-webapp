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
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from xgboost import XGBClassifier


# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="VerifyIt Nigeria - Investment Scam Detector",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import NLTK components with better error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Try to download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
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

# Enhanced stopwords (including common Nigerian terms)
ENHANCED_STOPWORDS = set([
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
    """Get stopwords - use NLTK if available, otherwise fallback"""
    if nltk_available:
        try:
            return set(stopwords.words('english'))
        except:
            return ENHANCED_STOPWORDS
    return ENHANCED_STOPWORDS

def tokenize_text(text):
    """Tokenize text - use NLTK if available, otherwise fallback"""
    if nltk_available:
        try:
            return word_tokenize(text.lower())
        except:
            return re.findall(r'\b\w+\b', text.lower())
    return re.findall(r'\b\w+\b', text.lower())

# Simple in-memory database for deployment
class SimpleDatabase:
    def __init__(self):
        if 'detections' not in st.session_state:
            st.session_state.detections = []
        if 'feedback' not in st.session_state:
            st.session_state.feedback = []
    
    def save_detection(self, text, prediction, probability):
        detection_id = len(st.session_state.detections) + 1
        st.session_state.detections.append({
            'id': detection_id,
            'text': text,
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.now().isoformat()
        })
        return detection_id
    
    def save_feedback(self, detection_id, is_correct, feedback_text=""):
        st.session_state.feedback.append({
            'detection_id': detection_id,
            'is_correct': is_correct,
            'feedback_text': feedback_text
        })
    
    def get_feedback_stats(self):
        if not st.session_state.feedback:
            return {'total_feedback': 0, 'correct_rate': 0}
        
        total = len(st.session_state.feedback)
        correct = sum(1 for f in st.session_state.feedback if f['is_correct'])
        return {
            'total_feedback': total,
            'correct_rate': (correct / total) * 100 if total > 0 else 0
        }
    
    def get_recent_detections(self, limit=50):
        return st.session_state.detections[-limit:] if st.session_state.detections else []
    
    def close(self):
        pass

# Enhanced Investment Scam Detector with Training Data
import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Simple function to get basic English stopwords
def get_stopwords():
    return [
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'you', 'your', 'this', 'have',
        'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
    ]

class EnhancedScamDetector:
    def __init__(self):
        # File paths for pre-trained models
        self.vectorizer_path = r"C:\Users\Hp\OneDrive\Desktop\VerifyIt- clone\vectorizer.pkl"
        self.model_path = r"C:\Users\Hp\OneDrive\Desktop\VerifyIt- clone\model.pkl"
        
        # Try to load pre-trained models
        self.loaded_vectorizer = None
        self.loaded_model = None
        self.load_pretrained_models()
        
        # Comprehensive scam indicators
        self.scam_keywords = [
            # General investment scam keywords
            'guaranteed', 'returns', 'profit', 'risk-free', 'limited time',
            'act now', 'exclusive', 'secret', 'insider', 'get rich',
            'easy money', 'no risk', 'double your money', 'investment opportunity',
            'high returns', 'make money fast', 'financial freedom', 'urgent',
            'congratulations', 'selected', 'approved', 'winner', 'claim',
            'immediate', 'expires', 'deadline', 'act fast', 'limited offer',
            
            # Nigerian-specific keywords
            'forex trading', 'binary options', 'cryptocurrency mining',
            'ponzi', 'pyramid scheme', 'matrix', 'gifting circle',
            'mmm', 'ultimate cycler', 'twinkas', 'zarfund',
            'crowd1', 'longrich', 'aim global', 'organo gold',
            'bitcoin investment', 'cryptocurrency doubler',
            'oil and gas investment', 'real estate flipping',
            'importation business', 'dollar buy and sell',
            'cbn', 'sec', 'federal', 'government grant', 'loan fund',
            
            # Currency and payment terms
            'naira', '₦', 'western union', 'moneygram', 'mobile money',
            'paypal', 'bitcoin wallet', 'perfect money', 'swift alert',
            'bank alert', 'debit alert', 'credit alert', 'otp', 'pin',
            'bvn', 'account number', 'bank details', 'transfer',
            
            # Communication platforms
            'whatsapp', 'telegram', 'click link', 'visit website',
            'call now', 'text message', 'sms', 'email',
            
            # Scam phrases
            'join my team', 'be your own boss', 'financial breakthrough',
            'poverty is a choice', 'multiple streams of income',
            'residual income', 'passive income', 'network marketing',
            'affiliate marketing', 'digital marketing', 'online business',
            'registration fee', 'processing fee', 'verification fee',
            'activation fee', 'send money', 'pay now', 'deposit'
        ]
        
        # Training data for the ML model
        self.training_data = [
            # Scam examples (label = 1)
            ("CONGRATULATIONS! CBN has selected you for a special 10,000,000 naira federal road contract. To claim your award, send your bank account number now.", 1),
            ("ATTENTION: Your request for a ₦5,000,000 CBN intervention grant has been approved. Reply with your full name and account details to receive the funds.", 1),
            ("SWIFT ALERT: A USD 10,000 transfer is pending in your account. We need your OTP/ATM PIN to verify. Send the code immediately to avoid cancellation.", 1),
            ("Join Crypto Bridge Exchange (CBEX) today! Deposit ₦100,000 and get ₦188,000 in 30 days (88% guaranteed ROI). No risk investment.", 1),
            ("New Horizon Investments promises 20% profit per week – guaranteed. Refer friends to double your earnings! Limited slots available.", 1),
            ("ForexKings Ltd: Guaranteed 15% profit daily with our FX signals. Invest ₦50,000 with our bot and watch it grow. WhatsApp us now.", 1),
            ("PUNISHER COIN presale is NOW OPEN! Buy $100 of $PUN tokens and earn 500% returns on launch. Limited offer.", 1),
            ("Helping Hand Finance – Double Your Money! Send ₦1,000 today and get ₦2,000 back in 24 hours. Trusted by thousands of Nigerians.", 1),
            ("CryptoMasterFX: Turn $100 into $10,000 in 10 days – instant returns, zero risk. Message us on WhatsApp to start!", 1),
            ("BANK ALERT: Your account ***1234 has been credited with ₦250,000. Please log in via our portal and confirm the transaction.", 1),
            ("Make $5000 weekly working from home! No experience needed. Send $200 registration fee to get started.", 1),
            ("Bitcoin doubler! Send 0.1 BTC and get 0.2 BTC back within 24 hours. 100% guarantee!", 1),
            ("URGENT: Nigerian prince needs your help transferring $50 million. You get 30% commission.", 1),
            ("Forex trading signals group! Join for $100 and earn $1000 daily. Limited time offer!", 1),
            ("Government stimulus check approved! Click here to claim your $2000 payment now.", 1),
            
            # Legitimate examples (label = 0)
            ("Thank you for your interest in our savings account. The current interest rate is 5% per annum. Visit our branch for more information.", 0),
            ("ABC Investment Ltd (SEC registered) offers diversified portfolio management. Minimum investment ₦500,000. Past performance doesn't guarantee future results.", 0),
            ("Our fixed deposit account offers 8% annual interest. Terms and conditions apply. Visit any of our branches nationwide.", 0),
            ("Investment Advisory: Market volatility may affect returns. Please consult our licensed financial advisors before investing.", 0),
            ("Monthly statement: Your mutual fund investment has gained 3% this month. Remember, investments can go up or down.", 0),
            ("Treasury Bills auction: Current rates between 6-9% annually. Minimum investment ₦10,000. Contact our investment team.", 0),
            ("Real estate investment opportunity in Lekki. Expected ROI 12-15% over 5 years. Due diligence required.", 0),
            ("Our pension fund has delivered consistent returns over the past decade. Contributions are tax-deductible.", 0),
            ("Stock market update: Your portfolio is up 2% this quarter. Diversification remains key to managing risk.", 0),
            ("Government bond auction: 10-year bonds at 11% yield. Backed by the full faith and credit of the government.", 0),
            ("Agricultural investment fund: Supporting Nigerian farmers with expected 10% annual returns. Risk disclosure available.", 0),
            ("Insurance-linked investment: Combine protection with growth potential. Speak to our certified agents.", 0),
            ("Our research team recommends balanced portfolio allocation. Schedule a free consultation with our advisors.", 0),
            ("Quarterly dividend payment of ₦50 per share has been credited to your account. Thank you for investing with us.", 0),
            ("Market analysis: Oil prices may impact energy sector investments. Consider rebalancing your portfolio.", 0),
            ("Our mutual fund has outperformed the market by 2% this year. Past performance is not indicative of future results.", 0),
            ("Corporate bond offering: 12% coupon rate, 5-year maturity. Credit rating: A-. Prospectus available.", 0),
            ("REIT investment: Own shares in commercial real estate. Quarterly distributions expected.", 0),
            ("Our financial planning service helps you achieve your long-term goals. Schedule a meeting today.", 0),
        ]
        
        # Weight multipliers for specific terms
        self.weight_multipliers = {
            'guaranteed': 3.0,
            'risk-free': 3.0,
            'double your money': 3.0,
            'get rich quick': 3.0,
            'ponzi': 4.0,
            'pyramid': 3.5,
            'mmm': 4.0,
            'bitcoin doubler': 4.0,
            'cbn': 2.5,
            'sec': 2.0,
            'congratulations': 2.0,
            'selected': 2.0,
            'approved': 2.0,
            'urgent': 2.0,
            'act now': 2.5,
            'limited time': 2.0,
            'whatsapp': 1.5,
            'telegram': 1.5,
            'otp': 3.0,
            'pin': 3.0,
            'bank alert': 2.5,
            'swift alert': 3.0,
            'registration fee': 2.5,
            'processing fee': 2.5,
            'verification fee': 2.5,
            'naira': 1.2,
            'forex': 2.0,
            'crypto': 1.5,
            'investment': 0.5,
            'returns': 0.5,
            'profit': 0.5,
        }
        
        # Initialize model variables
        self.model = None
        self.vectorizer = None
        
        # Use pre-trained models if available, otherwise train new ones
        if self.loaded_vectorizer is not None and self.loaded_model is not None:
            self.vectorizer = self.loaded_vectorizer
            self.model = self.loaded_model
            print("✅ Pre-trained models loaded successfully!")
        else:
            self.train_model()
    
    def load_pretrained_models(self):
        """Load pre-trained models if they exist"""
        try:
            if os.path.exists(self.vectorizer_path):
                with open(self.vectorizer_path, 'rb') as f:
                    self.loaded_vectorizer = pickle.load(f)
                print("✅ Vectorizer loaded from file")
            
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.loaded_model = pickle.load(f)
                print("✅ Model loaded from file")
                
        except Exception as e:
            print(f"⚠️ Could not load pre-trained models: {str(e)}")
            self.loaded_vectorizer = None
            self.loaded_model = None
    
    def save_models(self):
        """Save trained models to files"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.vectorizer_path), exist_ok=True)
            
            # Save vectorizer
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            print("✅ Models saved successfully!")
            
        except Exception as e:
            print(f"⚠️ Could not save models: {str(e)}")
    
    def train_model(self):
        """Train the ML model with our training data"""
        try:
            # Prepare training data
            texts = [item[0] for item in self.training_data]
            labels = [item[1] for item in self.training_data]
            
            # Create TF-IDF vectorizer with custom stopwords
            custom_stopwords = get_stopwords()
            
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=custom_stopwords,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Transform texts to vectors
            X = self.vectorizer.fit_transform(texts)
            y = np.array(labels)
            
            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Logistic Regression model (more stable for deployment)
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            print(f"✅ Model trained successfully!")
            print(f"Training accuracy: {train_accuracy:.3f}")
            print(f"Testing accuracy: {test_accuracy:.3f}")
            
            # Save the trained models
            self.save_models()
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            self.model = None
            self.vectorizer = None
    
    def keyword_based_prediction(self, text):
        """Fallback keyword-based prediction"""
        text_lower = text.lower()
        scam_score = 0
        
        # Count scam indicators with weighting
        for keyword in self.scam_keywords:
            if keyword in text_lower:
                weight = self.weight_multipliers.get(keyword, 1.0)
                scam_score += weight
        
        # Additional pattern checks
        if re.search(r'[\d,]+\s*(naira|₦|\$|USD)', text_lower):
            scam_score += 1.0
        
        if re.search(r'(guaranteed|100%|no risk)', text_lower):
            scam_score += 2.0
        
        if re.search(r'(whatsapp|telegram|click|link)', text_lower):
            scam_score += 1.0
        
        if re.search(r'(otp|pin|password|account)', text_lower):
            scam_score += 2.0
        
        # Calculate probability
        probability = min(scam_score / 10.0, 0.95)
        
        # Determine prediction
        if probability > 0.5:
            return "SCAM", probability
        else:
            return "LEGITIMATE", 1 - probability
    
    def predict(self, text):
        """Make prediction using ML model or fallback to keyword-based"""
        if self.model is not None and self.vectorizer is not None:
            try:
                # Transform text using vectorizer
                text_vector = self.vectorizer.transform([text])
                
                # Get prediction probability
                prediction_proba = self.model.predict_proba(text_vector)[0]
                scam_prob = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
                
                # Combine ML prediction with keyword-based scoring for better accuracy
                keyword_pred, keyword_prob = self.keyword_based_prediction(text)
                
                # Weighted combination (70% ML, 30% keyword-based)
                combined_prob = (0.7 * scam_prob) + (0.3 * (keyword_prob if keyword_pred == "SCAM" else 1 - keyword_prob))
                
                # Determine final prediction
                if combined_prob > 0.5:
                    return "SCAM", combined_prob
                else:
                    return "LEGITIMATE", 1 - combined_prob
                    
            except Exception as e:
                print(f"⚠️ ML model prediction failed: {str(e)}")
                return self.keyword_based_prediction(text)
        else:
            return self.keyword_based_prediction(text)

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = EnhancedScamDetector()
    
    # Test with some examples
    test_messages = [
        "CONGRATULATIONS! You've won ₦1,000,000! Send your bank details to claim now!",
        "Our mutual fund offers 8% annual returns. Past performance doesn't guarantee future results.",
        "Bitcoin doubler! Send 0.1 BTC get 0.2 BTC back in 24 hours. 100% guaranteed!",
        "Thank you for your interest in our savings account. Visit our branch for more information."
    ]
    
    print("\n" + "="*60)
    print("TESTING SCAM DETECTOR")
    print("="*60)
    
    for i, message in enumerate(test_messages, 1):
        prediction, confidence = detector.predict(message)
        print(f"\nTest {i}:")
        print(f"Message: {message[:60]}...")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print("-" * 60)

# Initialize session state
def initialize_session_state():
    if 'db' not in st.session_state:
        st.session_state.db = SimpleDatabase()
    
    if 'model' not in st.session_state:
        with st.spinner("🤖 Initializing AI model..."):
            st.session_state.model = EnhancedScamDetector()
            if st.session_state.model.model is not None:
                st.success("✅ Enhanced ML model loaded successfully!")
            else:
                st.warning("⚠️ Using keyword-based detection")
    
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []

# Initialize everything
initialize_session_state()

# Custom CSS for styling with theme responsiveness
st.markdown("""
<style>
    /* Main header - responsive to theme */
    .main-header {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 50%, #228B22 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white !important;
        text-align: center;
        border: 2px solid #228B22;
    }
    
    /* Ensure header text is always white regardless of theme */
    .main-header h1, .main-header h2, .main-header h3, .main-header p {
        color: white !important;
    }
    
    /* Scam prediction - always white text on red background */
    .prediction-scam {
        background: linear-gradient(135deg, #DC143C 0%, #FF4444 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Ensure scam prediction text is always white */
    .prediction-scam h1, .prediction-scam h2, .prediction-scam h3, .prediction-scam p {
        color: white !important;
    }
    
    /* Safe prediction - always white text on green background */
    .prediction-safe {
        background: linear-gradient(135deg, #228B22 0%, #32CD32 100%);
        color: white !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Ensure safe prediction text is always white */
    .prediction-safe h1, .prediction-safe h2, .prediction-safe h3, .prediction-safe p {
        color: white !important;
    }
    
    /* Sidebar info - responsive to theme with black text */
    .sidebar-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #228B22;
        color: black !important;
    }
    
    /* Ensure sidebar info text is always black */
    .sidebar-info h1, .sidebar-info h2, .sidebar-info h3, .sidebar-info p, .sidebar-info li {
        color: black !important;
    }
    
    /* Sidebar disclaimer - responsive to theme with black text */
    .sidebar-disclaimer {
        background-color: #ffe4e1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #dc143c;
        color: black !important;
    }
    
    /* Ensure sidebar disclaimer text is always black */
    .sidebar-disclaimer h1, .sidebar-disclaimer h2, .sidebar-disclaimer h3, .sidebar-disclaimer p, .sidebar-disclaimer li {
        color: black !important;
    }
    
    /* Dark theme adaptations */
    @media (prefers-color-scheme: dark) {
        .sidebar-info {
            background-color: #e6f3ff;
            color: black !important;
        }
        
        .sidebar-disclaimer {
            background-color: #ffd6d6;
            color: black !important;
        }
    }
    
    /* Streamlit dark theme specific overrides */
    .stApp[data-theme="dark"] .sidebar-info {
        background-color: #e6f3ff;
        color: black !important;
    }
    
    .stApp[data-theme="dark"] .sidebar-disclaimer {
        background-color: #ffd6d6;
        color: black !important;
    }
    
    /* Additional responsive text classes for general use */
    .responsive-text {
        color: var(--text-color);
    }
    
    /* Light theme text */
    .stApp[data-theme="light"] .responsive-text {
        color: #000000;
    }
    
    /* Dark theme text */
    .stApp[data-theme="dark"] .responsive-text {
        color: #ffffff;
    }
    
    /* Fallback for browsers that don't support CSS variables */
    @media (prefers-color-scheme: light) {
        .responsive-text {
            color: #000000;
        }
    }
    
    @media (prefers-color-scheme: dark) {
        .responsive-text {
            color: #ffffff;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🇳🇬 VerifyIt Nigeria</h1>
    <h3>AI-Powered Investment Scam Detector</h3>
    <p>Protecting Nigerian Investors with Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🇳🇬 Navigation")
    page = st.selectbox(
        "Choose a page:",
        ["🔍 Scam Detection", "📈 Analytics", "📋 History", "ℹ️ About"]
    )
    
    st.markdown("""
    <div class="sidebar-info">
        <h4>🚨 Nigerian Scam Warning Signs</h4>
        <ul>
            <li>Forex trading guarantees</li>
            <li>MMM-style schemes</li>
            <li>Bitcoin doublers</li>
            <li>WhatsApp/Telegram recruitment</li>
            <li>Naira multiplication promises</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-disclaimer">
        <h4>⚠️ IMPORTANT DISCLAIMER</h4>
        <p><strong>This tool provides risk assessment, NOT financial advice!</strong></p>
        <p>Always conduct independent research and consult licensed financial advisors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    stats = st.session_state.db.get_feedback_stats()
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Scans", len(st.session_state.detection_history))
    with col2:
        st.metric("Accuracy", f"{stats['correct_rate']:.1f}%")

# Main content based on page selection
if page == "🔍 Scam Detection":
    st.markdown("## 🔍 Investment Scam Detection")
    
    # Text input for analysis
    user_input = st.text_area(
        "📝 Paste the investment offer or message you want to analyze:",
        height=150,
        placeholder="Example: Join our WhatsApp group for guaranteed forex profits! 100% returns in 7 days..."
    )
    
    if st.button("🔍 Analyze for Scams"):
        if user_input.strip():
            with st.spinner("🤖 Analyzing text for scam indicators..."):
                # Get prediction from model
                prediction, probability = st.session_state.model.predict(user_input)
                
                # Save detection
                detection_id = st.session_state.db.save_detection(
                    user_input, prediction, probability
                )
                
                # Add to history
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
                    
                    st.error("⚠️ **WARNING**: This message shows multiple red flags commonly associated with investment scams.")
                    
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
                    if st.button("👍 Yes, accurate", key=f"yes_{detection_id}"):
                        st.session_state.db.save_feedback(detection_id, True)
                        st.success("Thank you for your feedback!")
                
                with col2:
                    if st.button("👎 No, incorrect", key=f"no_{detection_id}"):
                        st.session_state.db.save_feedback(detection_id, False)
                        st.warning("Thank you for the feedback. We'll use this to improve the system.")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

elif page == "📈 Analytics":
    st.markdown("## 📈 Analytics Dashboard")
    
    if st.session_state.detection_history:
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
        
        # Simple chart
        fig = px.pie(
            values=df['prediction'].value_counts().values,
            names=df['prediction'].value_counts().index,
            title="Prediction Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📊 No analysis data available yet. Start by analyzing some investment offers!")

elif page == "📋 History":
    st.markdown("## 📋 Detection History")
    
    if st.session_state.detection_history:
        df = pd.DataFrame(st.session_state.detection_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            df[['timestamp', 'text', 'prediction', 'probability']],
            use_container_width=True
        )
    else:
        st.info("📋 No detection history available yet.")

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About VerifyIt Nigeria")
    
    st.markdown("""
    ### 🎯 Mission
    VerifyIt Nigeria is an AI-powered investment scam detection system designed to protect Nigerian investors from fraudulent schemes.
    
    ### 🧠 How It Works
    Our system uses machine learning and natural language processing to identify potential scams by analyzing:
    - Keyword patterns common in scams
    - Nigerian-specific fraud indicators
    - Investment offer language
    - Risk assessment scoring
    
    ### 📊 Technology
    - **Machine Learning**: Scikit-learn models
    - **NLP**: Text analysis and feature extraction
    - **Frontend**: Streamlit framework
    - **Database**: Session-based storage
    
    ### ⚠️ Disclaimer
    This tool provides risk assessment, not financial advice. Always:
    - Conduct independent research
    - Consult licensed financial advisors
    - Verify with Nigerian regulatory authorities
    - Use your own judgment
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="info-footer">
    <p>🇳🇬 <strong>VerifyIt Nigeria</strong> - Protecting Nigerian Investors with AI</p>
    <p>For educational and research purposes | Always verify investments independently</p>
</div>
""", unsafe_allow_html=True)
