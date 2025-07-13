# VerifyIt Nigeria - Full-Stack Implementation Documentation

## 📋 Project Overview

**VerifyIt Nigeria** is an AI-powered investment scam detection system designed to protect Nigerian investors from fraudulent investment schemes. This documentation covers the complete full-stack implementation including the original Python/Streamlit application, the new FastAPI backend, and the modern Next.js frontend.

### 🎓 Research Attribution

- **Author**: Ononneobazi Aquah (Computer Science Student)
- **Supervisor**: Prof. Moses Adah Agana (Professor of Computer Science)
- **Institution**: University of Calabar, Nigeria
- **Project Type**: Undergraduate Research Project

---

## 🏗️ Architecture Overview

The project has been transformed from a single Streamlit application into a modern full-stack architecture:

```
VerifyIt-webapp/
├── Codes/                    # Original Python implementation
│   ├── app.py               # Original Streamlit app (1,138 lines)
│   ├── database.py          # SQLite database management
│   ├── model.pkl            # Pre-trained ML model
│   ├── vectorizer.pkl       # TF-IDF vectorizer
│   └── scam_detector.db     # SQLite database file
├── api/                     # FastAPI Backend
│   └── main.py              # REST API server
├── verifyit-frontend/       # Next.js Frontend
│   └── src/app/page.tsx     # Main dashboard component
└── streamlit_app.py         # Root Streamlit app (duplicate)
```

---

## 🔍 Original System Analysis

### What Was Discovered

**1. Nigerian-Specific Scam Detection Engine**

- 50+ Nigerian-specific scam keywords including "MMM", "forex trading", "ponzi", "oga", "naira"
- Weighted detection system with multipliers (e.g., "MMM" = 3.0x weight, "forex" = 2.0x weight)
- Context-aware analysis for WhatsApp/Telegram schemes
- Integration with Nigerian regulatory warnings

**2. Machine Learning Components**

- Pre-trained XGBoost classifier (`model.pkl`)
- TF-IDF vectorizer for text processing (`vectorizer.pkl`)
- Fallback keyword-based detection when ML models unavailable
- Accuracy: ~94.2%, Precision: ~91.8%, Recall: ~96.3%

**3. Database System**

- SQLite database (`scam_detector.db`) for detection history
- User feedback system for continuous learning
- Performance analytics and reporting

**4. Nigerian Emergency Contacts**

- EFCC: 0800-CALL-EFCC (0800-2255-3322)
- SEC Nigeria: +234-9-461-0000
- Nigeria Police: 199 (Emergency)
- CBN: +234-700-2255-226

---

## 🚀 FastAPI Backend Implementation

### File: `api/main.py`

**Key Features Implemented:**

1. **Nigerian Scam Detector Class**

   ```python
   class NigerianScamDetector:
       def __init__(self):
           self.nigerian_scam_keywords = [
               'guaranteed', 'returns', 'forex trading', 'ponzi',
               'mmm', 'bitcoin investment', 'naira', '₦', 'lagos', 'oga'
           ]
           self.nigerian_weight_multipliers = {
               'forex': 2.0, 'mmm': 3.0, 'ponzi': 3.0, 'pyramid': 2.5
           }
   ```

2. **REST API Endpoints**

   - `GET /` - API information and attribution
   - `POST /analyze` - Text analysis for scam detection
   - `GET /stats` - Application statistics
   - `GET /emergency-contacts` - Nigerian emergency contacts
   - `GET /health` - Health check endpoint

3. **Analysis Response Format**
   ```json
   {
     "prediction": "SCAM" | "LEGITIMATE",
     "probability": 0.87,
     "risk_level": "HIGH" | "MEDIUM" | "LOW",
     "confidence": "High" | "Medium",
     "indicators": ["Guaranteed Returns", "Forex", "Naira"]
   }
   ```

### Installation & Setup

```bash
# Install dependencies
cd api
python3 -m pip install fastapi uvicorn

# Start the server
python3 -m uvicorn main:app --reload --port 8000
```

**Backend Status**: ✅ **FULLY FUNCTIONAL**

- Server runs on `http://localhost:8000`
- All endpoints working
- Nigerian scam detection active
- CORS configured for frontend integration

---

## 🎨 Next.js Frontend Implementation

### File: `verifyit-frontend/src/app/page.tsx`

**Architecture**: Modern dashboard built with Shadcn UI components

**Key Features Implemented:**

1. **Tab-Based Navigation**

   - Dashboard: Key metrics and recent activity
   - Scan Message: Real-time investment message analysis
   - Analytics: Performance metrics and trends
   - Education: Nigerian scam awareness content
   - Report Scam: Emergency contacts and reporting

2. **Real-Time API Integration**

   ```typescript
   const analyzeText = async () => {
     const response = await fetch("http://localhost:8000/analyze", {
       method: "POST",
       headers: { "Content-Type": "application/json" },
       body: JSON.stringify({ text: analysisText }),
     });
     const result = await response.json();
     setAnalysisResult(result);
   };
   ```

3. **Dynamic Data Loading**

   - Stats fetched from `/stats` endpoint
   - Emergency contacts from `/emergency-contacts` endpoint
   - Live scam analysis results
   - Responsive design with loading states

4. **Nigerian Context Integration**
   - Country-specific badge and branding
   - Local emergency contacts display
   - Nigerian scam education content
   - Naira currency and local terminology

### Installation & Setup

```bash
# Navigate to frontend
cd verifyit-frontend

# Install dependencies (if needed)
npm install

# Start development server
npm run dev
```

**Frontend Status**: ✅ **FULLY FUNCTIONAL**

- Modern Shadcn UI implementation
- Real-time backend integration
- Responsive design
- Nigerian context preserved

---

## 🔗 API Convergence & Integration

### How Frontend & Backend Connect

1. **CORS Configuration**

   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],  # Next.js frontend
       allow_credentials=True
   )
   ```

2. **Data Flow**

   ```
   User Input → Next.js Frontend → FastAPI Backend → ML Analysis → Response → UI Update
   ```

3. **Real-Time Features**

   - Live scam detection analysis
   - Dynamic statistics updates
   - Emergency contacts loading
   - Interactive feedback system

4. **Error Handling**
   - Graceful API failure handling
   - Loading states during analysis
   - Fallback to offline mode when needed

---

## 🎯 Current Functionality Status

### ✅ What Works (Fully Functional)

**Backend (FastAPI)**

- ✅ Nigerian scam detection engine
- ✅ Keyword-based analysis with weighting
- ✅ REST API endpoints
- ✅ CORS configuration
- ✅ Health checks
- ✅ Emergency contacts API
- ✅ Statistics endpoint

**Frontend (Next.js)**

- ✅ Modern dashboard interface
- ✅ Real-time text analysis
- ✅ Dynamic data loading
- ✅ Tab-based navigation
- ✅ Responsive design
- ✅ Loading states and error handling
- ✅ Nigerian context preservation

**Integration**

- ✅ Frontend-backend communication
- ✅ Live scam detection
- ✅ API data synchronization

### ⚠️ What's Partially Working

**Machine Learning Models**

- ⚠️ Pre-trained models exist (`model.pkl`, `vectorizer.pkl`) but not fully integrated in FastAPI
- ⚠️ Currently using keyword-based detection as primary method
- ⚠️ Database integration available but using fallback system

**Advanced Features**

- ⚠️ User feedback system (backend ready, frontend not fully connected)
- ⚠️ Detection history (database exists, UI not fully implemented)
- ⚠️ Analytics trends (mock data currently used)

### ❌ What Doesn't Work Yet

**Missing Integrations**

- ❌ Full ML model integration in FastAPI
- ❌ User authentication system
- ❌ Persistent detection history in frontend
- ❌ Advanced analytics dashboards
- ❌ Export functionality
- ❌ Multi-language support

---

## 🔧 How to Run Everything

### Complete Setup Process

**1. Start the FastAPI Backend**

```bash
cd api
python3 -m uvicorn main:app --reload --port 8000
```

- Runs on: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

**2. Start the Next.js Frontend**

```bash
cd verifyit-frontend
npm run dev
```

- Runs on: `http://localhost:3000`
- Connects automatically to backend

**3. Test the Original Streamlit App (Optional)**

```bash
cd Codes
streamlit run app.py --server.port 8501
```

- Runs on: `http://localhost:8501`
- Original full-featured application

### Troubleshooting

**Port Conflicts**

- If port 8000 is in use: `lsof -i :8000` then `kill <PID>`
- Alternative: Use different port `--port 8001`

**Dependencies**

- FastAPI: `pip install fastapi uvicorn`
- Next.js: All dependencies included in `package.json`

---

## 📝 Where to Make Updates

### Backend Updates (FastAPI - `api/main.py`)

**For Nigerian Scam Detection:**

```python
# Add new keywords
self.nigerian_scam_keywords.extend([
    'new_scam_term', 'another_indicator'
])

# Adjust weights
self.nigerian_weight_multipliers['new_term'] = 2.5
```

**For New API Endpoints:**

```python
@app.get("/new-endpoint")
async def new_feature():
    return {"message": "New feature"}
```

**For ML Model Integration:**

```python
# In NigerianScamDetector class
def load_model(self):
    with open('../Codes/model.pkl', 'rb') as f:
        self.model = pickle.load(f)
    with open('../Codes/vectorizer.pkl', 'rb') as f:
        self.vectorizer = pickle.load(f)
```

### Frontend Updates (Next.js - `verifyit-frontend/src/app/page.tsx`)

**For UI Changes:**

- Update Shadcn UI components
- Modify tab content and layout
- Add new dashboard widgets

**For New API Integrations:**

```typescript
const fetchNewData = async () => {
  const response = await fetch("http://localhost:8000/new-endpoint");
  const data = await response.json();
  setNewData(data);
};
```

**For Nigerian Context:**

- Update emergency contacts
- Add new scam education content
- Modify local terminology

### Original System Updates (`Codes/` directory)

**Database Schema Changes:**

- Modify `database.py` for new tables
- Update `app.py` for new Streamlit features

**ML Model Improvements:**

- Retrain models with new data
- Update `model.pkl` and `vectorizer.pkl`
- Modify detection algorithms

---

## 🔄 Update Strategy Recommendations

### Recommended Approach: **Dual Development**

1. **Use FastAPI + Next.js for new features**

   - Modern architecture
   - Better scalability
   - API-first approach
   - Professional UI/UX

2. **Maintain original Streamlit app for research**
   - Full ML model integration
   - Academic research features
   - Rapid prototyping
   - Data analysis tools

### API vs Original System Updates

**Update APIs when:**

- Adding new endpoints
- Improving user interface
- Scaling for production
- Integrating with external systems

**Update original system when:**

- Improving ML models
- Academic research needs
- Testing new algorithms
- Data analysis requirements

---

## 🚀 Production Deployment Recommendations

### Backend Deployment

- Use Gunicorn with Uvicorn workers
- Deploy on cloud platforms (AWS, GCP, Azure)
- Set up proper environment variables
- Configure production database (PostgreSQL)

### Frontend Deployment

- Deploy on Vercel (recommended for Next.js)
- Configure environment variables for API endpoints
- Set up proper domain and SSL

### Security Considerations

- Add authentication/authorization
- Rate limiting for API endpoints
- Input validation and sanitization
- HTTPS enforcement

---

## 📞 Support & Contact

**Research Team:**

- **Student**: Ononneobazi Aquah
- **Supervisor**: Prof. Moses Adah Agana
- **Institution**: University of Calabar, Computer Science Department

**Technical Issues:**

- Check logs in terminal output
- Test API endpoints at `http://localhost:8000/docs`
- Verify CORS settings for frontend integration

**Emergency Contacts (for scam reporting):**

- EFCC: 0800-CALL-EFCC
- SEC Nigeria: +234-9-461-0000
- Nigeria Police: 199

---

## 📈 Future Enhancements

1. **Full ML Integration**: Complete integration of pre-trained models
2. **User Authentication**: Login system and user profiles
3. **Advanced Analytics**: Real-time detection trends and insights
4. **Mobile App**: React Native or Flutter implementation
5. **Multi-language Support**: Hausa, Yoruba, Igbo language detection
6. **Regulatory Integration**: Real-time updates from SEC/EFCC databases

---

_This documentation covers the complete full-stack implementation of VerifyIt Nigeria. For specific technical questions, refer to the code comments and API documentation._
