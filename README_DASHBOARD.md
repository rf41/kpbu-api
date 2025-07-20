# 🏗️ KPBU API Test Dashboard

## 📋 Quick Start Guide

### 1. Install Dependencies
```bash
# Install dashboard requirements
pip install -r dashboard_requirements.txt

# Or install manually
pip install streamlit requests
```

### 2. Start API Server
```bash
# Terminal 1: Start the API server
python3 api_matchmaker.py
```

### 3. Start Dashboard
```bash
# Terminal 2: Start the dashboard
streamlit run test_dashboard.py
```

### 4. Access Dashboard
Open browser: `http://localhost:8501`

## 🎯 Dashboard Features

### **Left Top: Investor Profile**
- Fill out investor risk profile
- Get project recommendations
- Save profile for chat integration

### **Left Bottom: Risk Prediction**
- Test project risk prediction
- Input project details
- Get AI-powered risk assessment

### **Right: AI Chatbot**
- Investment consultation chat
- Requires filled investor profile
- Persuasive investment guidance
- Session-based conversations

### **Footer Controls**
- Dataset statistics
- Gemini API diagnostics
- Session cleanup utilities

## 🚀 Usage Workflow

1. **Fill Investor Profile** → Choose risk tolerance, sectors, horizon, investment size
2. **Click "Save Profile"** → Get personalized project recommendations
3. **Test Risk Prediction** → Try predicting risk for custom projects  
4. **Start Chat** → Begin AI investment consultation
5. **Ask Questions** → Get persuasive investment advice
6. **Test API Features** → Use footer buttons for diagnostics

## 🎨 Design Features

- **Minimalist Design**: Clean, professional interface
- **Responsive Layout**: Two-column layout optimized for speed
- **Real-time Updates**: Live API connection status
- **Interactive Forms**: Streamlined user input
- **Session Management**: Persistent chat sessions
- **Error Handling**: Graceful error messages
- **Fast Loading**: Optimized for performance

## 🔧 Troubleshooting

### API Connection Issues
```bash
# Check if API is running
curl http://localhost:8000/health

# Check API logs
tail -f api_server.log
```

### Streamlit Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart dashboard
streamlit run test_dashboard.py --server.port 8501
```

### Gemini API Issues
```bash
# Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# Test Gemini endpoint
curl -X GET "http://localhost:8000/test-gemini" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

## 📊 API Endpoints Tested

- `POST /match` - Investor matching
- `POST /predict` - Risk prediction  
- `POST /chat/start` - Start chat session
- `POST /chat/message` - Send chat message
- `GET /dataset/stats` - Dataset statistics
- `GET /test-gemini` - Gemini API test
- `DELETE /chat/cleanup` - Session cleanup

## 🎪 Demo Data

### Sample Investor Profiles:
- **Konservatif**: Air & Sanitasi, Kesehatan (Low risk)
- **Moderat**: Jalan & Jembatan, Infrastruktur (Balanced)
- **Agresif**: Energi Terbarukan, Teknologi (High return)

### Sample Projects:
- **Jalan Tol**: High investment, medium risk
- **PDAM**: Medium investment, low risk  
- **PLTS**: Large investment, high risk
- **Rumah Sakit**: Small investment, low risk

## 🏆 Performance Optimizations

- **Streamlit Session State**: Efficient state management
- **API Connection Pooling**: Reused HTTP connections
- **Lazy Loading**: Components load on demand
- **Minimal CSS**: Lightweight styling
- **Fast Rendering**: Optimized UI updates

This dashboard provides a complete testing environment for all KPBU API features in a single, fast-loading interface! 🚀
