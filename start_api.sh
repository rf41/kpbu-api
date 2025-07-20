#!/bin/bash

echo "🚀 Starting KPBU Investor Matchmaker API..."
echo "============================================="

# Check if data file exists
if [ ! -f "model/data/data_kpbu.csv" ]; then
    echo "❌ Error: Data file not found at model/data/data_kpbu.csv"
    echo "Please ensure the data file exists before running the API."
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies ready!"
echo ""
echo "🌐 Starting API server..."
echo "   - Local: http://localhost:8000"
echo "   - Docs: http://localhost:8000/docs"
echo "   - Auth Token: kpbu-matchmaker-2025"
echo ""
echo "Press Ctrl+C to stop the server"
echo "============================================="

# Start the API
python api_matchmaker.py
