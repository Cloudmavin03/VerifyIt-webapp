import joblib
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class ScamDetector:
    def __init__(self, model_path="scam_detector_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
        """Initialize the scam detector with pretrained model and vectorizer"""
        try:
            # Load the model and vectorizer
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Download required NLTK resources if not already downloaded
            try:
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                print("Downloading required NLTK resources...")
                nltk.download('stopwords')
                nltk.download('wordnet')
                
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            print("Scam detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing scam detector: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text for prediction"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, numbers, URLs, etc.
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Lemmatize tokens
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Rejoin tokens
        processed_text = ' '.join(tokens)
        return processed_text
    
    def predict(self, text):
        """Predict whether the text is a scam or not"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Transform text to TF-IDF features
            features = self.vectorizer.transform([processed_text])
            
            # Make prediction
            prediction_proba = self.model.predict_proba(features)[0]
            prediction_class = self.model.predict(features)[0]
            
            # Get confidence score (probability of predicted class)
            scam_probability = prediction_proba[1] if prediction_class == 1 else prediction_proba[0]
            
            # Determine prediction label
            prediction_label = "Scam" if prediction_class == 1 else "Legitimate"
            
            return {
                "prediction": prediction_label,
                "probability": float(scam_probability),
                "processed_text": processed_text
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {
                "prediction": "Error",
                "probability": 0.0,
                "error": str(e)
            }
    
    def get_key_indicators(self, text, top_n=5):
        """Extract key indicators that influenced the prediction"""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Get feature names from vectorizer
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Transform text to get feature vector
            feature_vector = self.vectorizer.transform([processed_text])
            
            # Get indices of non-zero features
            non_zero_indices = feature_vector.nonzero()[1]
            
            # Get feature names and values
            feature_values = [(feature_names[i], feature_vector[0, i]) for i in non_zero_indices]
            
            # Sort features by TF-IDF value (importance)
            sorted_features = sorted(feature_values, key=lambda x: x[1], reverse=True)
            
            # Return top N features
            return sorted_features[:top_n]
        except Exception as e:
            print(f"Error getting key indicators: {e}")
            return []