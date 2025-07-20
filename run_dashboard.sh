#!/bin/bash

# KPBU Dashboard Launcher Script
echo "🏗️ KPBU API Test Dashboard Launcher"
echo "=================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit..."
    pip install streamlit requests
fi

# Check if API server is running
echo "🔍 Checking API server status..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API server is running on http://localhost:8000"
else
    echo "⚠️  API server not detected. Please start it first:"
    echo "   python3 api_matchmaker.py"
    echo ""
    echo "🚀 Starting dashboard anyway..."
fi

echo ""
echo "🎯 Launching Test Dashboard..."
echo "   Dashboard URL: http://localhost:8501"
echo "   API URL: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "=================================="

# Launch Streamlit dashboard
streamlit run test_dashboard.py --server.port 8501 --server.address localhost
