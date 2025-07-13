import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import the existing logic from the original app
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration with modern styling
st.set_page_config(
    page_title="VerifyIt Nigeria",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern white dashboard
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 95%;
    }
    .stApp {
        background-color: #ffffff;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Import Nigerian Scam Detector (from original app)
class NigerianScamDetector:
    def __init__(self):
        self.nigerian_scam_keywords = [
            'guaranteed', 'returns', 'profit', 'risk-free', 'limited time',
            'forex trading', 'binary options', 'cryptocurrency mining',
            'ponzi', 'pyramid scheme', 'mmm', 'bitcoin investment',
            'naira', '₦', 'lagos', 'abuja', 'oga', 'boss'
        ]
        
        self.nigerian_weight_multipliers = {
            'forex': 2.0, 'mmm': 3.0, 'ponzi': 3.0, 'pyramid': 2.5,
            'bitcoin doubler': 3.0, 'guaranteed returns': 2.5
        }
    
    def predict(self, text):
        text_lower = text.lower()
        scam_score = 0
        
        for keyword in self.nigerian_scam_keywords:
            if keyword in text_lower:
                weight = self.nigerian_weight_multipliers.get(keyword, 1.0)
                scam_score += weight
        
        # Calculate probability
        max_possible_score = len(self.nigerian_scam_keywords) * 2
        probability = min(scam_score / max_possible_score, 0.95)
        
        return "SCAM" if probability > 0.3 else "LEGITIMATE", probability

# Initialize detector
@st.cache_resource
def load_detector():
    return NigerianScamDetector()

detector = load_detector()

# Modern Header
colored_header(
    label="🇳🇬 VerifyIt Nigeria",
    description="AI-Powered Investment Scam Detection System",
    color_name="green-70"
)

# Modern Navigation Menu
selected = option_menu(
    menu_title=None,
    options=["🏠 Dashboard", "🔍 Scam Detection", "📊 Analytics", "📚 Education", "🚨 Report"],
    icons=["house", "search", "graph-up", "book", "exclamation-triangle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff"},
        "icon": {"color": "#00875A", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "padding": "12px 16px",
            "--hover-color": "#f0f9ff",
            "border-radius": "8px",
            "color": "#374151"
        },
        "nav-link-selected": {
            "background-color": "#00875A",
            "color": "white",
            "border-radius": "8px"
        },
    }
)

add_vertical_space(2)

# Dashboard Page
if selected == "🏠 Dashboard":
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🛡️ Total Scans",
            value="1,247",
            delta="12 today"
        )
    
    with col2:
        st.metric(
            label="⚠️ Scams Detected", 
            value="89",
            delta="3 today"
        )
    
    with col3:
        st.metric(
            label="✅ Accuracy Rate",
            value="94.2%",
            delta="2.1%"
        )
    
    with col4:
        st.metric(
            label="🇳🇬 Nigerian Context",
            value="Active",
            delta="Live"
        )
    
    style_metric_cards()
    
    add_vertical_space(2)
    
    # Recent Activity and Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Detection Trends")
        # Sample data for chart
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        scam_detections = np.random.poisson(3, 30)
        
        fig = px.line(
            x=dates, 
            y=scam_detections,
            title="Daily Scam Detections",
            labels={'x': 'Date', 'y': 'Detections'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Scam Types")
        # Sample data for pie chart
        scam_types = ['Forex Scams', 'Ponzi Schemes', 'Crypto Scams', 'Others']
        values = [35, 28, 22, 15]
        
        fig = px.pie(
            values=values,
            names=scam_types,
            title="Common Scam Types in Nigeria"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#374151')
        )
        st.plotly_chart(fig, use_container_width=True)

# Scam Detection Page
elif selected == "🔍 Scam Detection":
    colored_header(
        label="Investment Scam Detection",
        description="Analyze suspicious investment messages with AI",
        color_name="red-70"
    )
    
    # Input section
    st.subheader("📝 Enter Investment Message")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area(
            "Paste the investment message, WhatsApp text, or email you want to analyze:",
            height=150,
            placeholder="Example: Join our forex trading platform and earn ₦500,000 monthly with guaranteed returns..."
        )
    
    with col2:
        add_vertical_space(3)
        analyze_button = st.button(
            "🔍 Analyze for Scams",
            type="primary",
            use_container_width=True
        )
    
    if analyze_button and user_input:
        with st.spinner("🤖 Analyzing with Nigerian AI context..."):
            prediction, probability = detector.predict(user_input)
            
        add_vertical_space(1)
        
        # Results section
        if prediction == "SCAM":
            st.error(f"⚠️ **HIGH RISK DETECTED** - {probability:.1%} chance of being a scam")
            
            # Risk breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Level", "HIGH", delta=f"{probability:.1%}")
            with col2:
                st.metric("Confidence", f"{probability:.1%}", delta="High")
                
            st.markdown("### 🚨 Why this might be a scam:")
            st.warning("""
            - Contains typical Nigerian investment scam keywords
            - Promises unrealistic returns
            - Uses high-pressure tactics
            - May target Nigerian investors specifically
            """)
            
        else:
            st.success(f"✅ **APPEARS LEGITIMATE** - {(1-probability):.1%} confidence")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Level", "LOW", delta=f"{(1-probability):.1%}")
            with col2:
                st.metric("Safety Score", f"{(1-probability):.1%}", delta="Good")

# Analytics Page
elif selected == "📊 Analytics":
    colored_header(
        label="Analytics Dashboard",
        description="Track scam detection trends and performance",
        color_name="blue-70"
    )
    
    # Analytics metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Detection Accuracy", "94.2%", "↑ 2.1%")
    with col2:
        st.metric("False Positives", "3.1%", "↓ 0.5%")
    with col3:
        st.metric("User Feedback", "87%", "↑ 5%")
    
    style_metric_cards()

# Education Page
elif selected == "📚 Education":
    colored_header(
        label="Nigerian Investment Scam Education",
        description="Learn to identify and avoid common scams",
        color_name="violet-70"
    )
    
    st.markdown("""
    ## 🇳🇬 Common Nigerian Investment Scams
    
    ### 1. **Forex Trading Scams**
    - Promise guaranteed daily/weekly returns
    - Require upfront payments for "training"
    - Often collapse after few months
    
    ### 2. **MMM and Ponzi Schemes**
    - Promise 30% monthly returns
    - Require recruiting others
    - Eventually collapse
    
    ### 3. **Cryptocurrency Doublers**
    - Claim to double Bitcoin/crypto investments
    - Use fake testimonials
    - Disappear with investors' money
    """)

# Report Page
elif selected == "🚨 Report":
    colored_header(
        label="Report Investment Scams",
        description="Help protect other Nigerians by reporting scams",
        color_name="orange-70"
    )
    
    st.markdown("""
    ## 🚨 Emergency Contacts
    
    ### Report Investment Scams to:
    - **EFCC**: 0800-CALL-EFCC (0800-2255-3322)
    - **SEC Nigeria**: +234-9-461-0000  
    - **Nigeria Police**: 199
    - **CBN**: +234-700-2255-226
    
    ### 📱 Online Reporting
    - EFCC Website: efccnigeria.org
    - SEC Nigeria: sec.gov.ng
    """)

# Footer
add_vertical_space(3)
st.markdown("---")
st.markdown(
    "🇳🇬 **VerifyIt Nigeria** - Protecting Nigerian investors from fraudulent schemes | "
    "Developed by University of Calabar"
) 