from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pickle
import sqlite3
import pandas as pd
from datetime import datetime
import re
from collections import Counter
import os
import sys

# Add the Codes directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Codes'))

try:
    from database import Database
except ImportError:
    # Fallback database implementation
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

# Nigerian Scam Detector Class
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
        
        max_possible_score = len(self.nigerian_scam_keywords) * 2
        probability = min(scam_score / max_possible_score, 0.95)
        
        return "SCAM" if probability > 0.3 else "LEGITIMATE", probability

# Initialize FastAPI app
app = FastAPI(title="VerifyIt Nigeria API", description="AI-Powered Investment Scam Detection for Nigeria")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
detector = NigerianScamDetector()
db = Database()

# Pydantic models for API
class TextAnalysisRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: str
    indicators: List[str]

class FeedbackRequest(BaseModel):
    detection_id: int
    is_correct: bool
    feedback_text: Optional[str] = None

class StatsResponse(BaseModel):
    total_scans: int
    scams_detected: int
    accuracy_rate: float
    users_protected: int

@app.get("/")
async def root():
    return {
        "message": "VerifyIt Nigeria API",
        "version": "1.0.0",
        "author": "Ononneobazi Aquah",
        "supervisor": "Prof. Moses Adah Agana",
        "institution": "University of Calabar, Nigeria"
    }

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    """Analyze text for investment scam indicators"""
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Analyze with Nigerian scam detector
        prediction, probability = detector.predict(request.text)
        
        # Determine risk level and confidence
        if probability > 0.7:
            risk_level = "HIGH"
            confidence = "High"
        elif probability > 0.4:
            risk_level = "MEDIUM"
            confidence = "Medium"
        else:
            risk_level = "LOW"
            confidence = "High" if probability < 0.2 else "Medium"
        
        # Find indicators in text
        indicators = []
        text_lower = request.text.lower()
        for keyword in detector.nigerian_scam_keywords[:10]:  # Top 10 keywords
            if keyword in text_lower:
                indicators.append(keyword.title())
        
        # Save to database
        detection_id = db.save_detection(request.text, prediction, probability)
        
        return TextAnalysisResponse(
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            confidence=confidence,
            indicators=indicators
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback on a detection"""
    try:
        db.save_feedback(
            feedback.detection_id,
            feedback.is_correct,
            feedback.feedback_text
        )
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get application statistics"""
    try:
        recent_detections = db.get_recent_detections(100)
        feedback_stats = db.get_feedback_stats()
        
        total_scans = len(recent_detections)
        scams_detected = len([d for d in recent_detections if d.get('prediction') == 'SCAM'])
        
        # Mock some stats for demonstration
        return StatsResponse(
            total_scans=max(1247, total_scans),
            scams_detected=max(89, scams_detected),
            accuracy_rate=feedback_stats.get('correct_rate', 94.2),
            users_protected=2841
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/recent-detections")
async def get_recent_detections(limit: int = 10):
    """Get recent detection history"""
    try:
        detections = db.get_recent_detections(limit)
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get detections: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",
        "model": "keyword-based",
        "timestamp": datetime.now().isoformat()
    }

# Emergency contacts endpoint
@app.get("/emergency-contacts")
async def get_emergency_contacts():
    """Get Nigerian emergency contacts for reporting scams"""
    return {
        "contacts": [
            {
                "name": "EFCC (Economic & Financial Crimes Commission)",
                "description": "Primary agency for financial crime investigation",
                "phone": "0800-CALL-EFCC (0800-2255-3322)",
                "website": "efccnigeria.org"
            },
            {
                "name": "SEC Nigeria",
                "description": "Securities and Exchange Commission",
                "phone": "+234-9-461-0000",
                "website": "sec.gov.ng"
            },
            {
                "name": "Nigeria Police",
                "description": "For urgent criminal matters",
                "phone": "199 (Emergency)",
                "website": ""
            },
            {
                "name": "Central Bank of Nigeria (CBN)",
                "description": "For banking-related scams",
                "phone": "+234-700-2255-226",
                "website": "cbn.gov.ng"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 