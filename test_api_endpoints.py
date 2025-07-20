#!/usr/bin/env python3
"""
Test script untuk memvalidasi semua endpoint KPBU API
Membantu debugging masalah "detail not found" dan endpoint connectivity
"""

import requests
import json
from datetime import datetime

# ===================== KONFIGURASI =====================
API_BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "kpbu-matchmaker-2025"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# ===================== HELPER FUNCTIONS =====================
def test_endpoint(method, endpoint, data=None, description=""):
    """Test an API endpoint and return formatted results"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {description}")
    print(f"📍 {method} {API_BASE_URL}{endpoint}")
    print(f"{'='*60}")
    
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "POST":
            response = requests.post(url, headers=HEADERS, json=data, timeout=15)
        elif method == "DELETE":
            response = requests.delete(url, headers=HEADERS, timeout=15)
        else:
            response = requests.get(url, headers=HEADERS, timeout=15)
        
        # Show request details
        if data:
            print(f"📤 REQUEST DATA:")
            print(json.dumps(data, indent=2))
            print()
        
        # Show response details
        print(f"📊 RESPONSE STATUS: {response.status_code}")
        print(f"📥 RESPONSE HEADERS: {dict(response.headers)}")
        print()
        
        try:
            response_json = response.json()
            print(f"📥 RESPONSE BODY:")
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            
            if response.status_code == 200:
                print(f"✅ SUCCESS: {description}")
                return True, response_json
            else:
                print(f"❌ FAILED: {description}")
                return False, response_json
                
        except json.JSONDecodeError:
            print(f"📥 RESPONSE BODY (Raw Text):")
            print(response.text)
            print(f"❌ FAILED: Invalid JSON response")
            return False, response.text
            
    except requests.exceptions.RequestException as e:
        print(f"🔌 CONNECTION ERROR: {str(e)}")
        return False, str(e)

def main():
    """Run all API endpoint tests"""
    print(f"🚀 KPBU API Endpoint Test Suite")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Base URL: {API_BASE_URL}")
    
    test_results = []
    
    # ===================== TEST 1: HEALTH CHECK =====================
    success, result = test_endpoint("GET", "/health", description="Health Check")
    test_results.append(("Health Check", success))
    
    # ===================== TEST 2: INVESTOR MATCHING =====================
    investor_profile = {
        "toleransi_risiko": "Moderat",
        "preferensi_sektor": [1, 3, 13],
        "horison_investasi": "Jangka Panjang",
        "ukuran_investasi": "Besar",
        "limit": 3
    }
    
    success, result = test_endpoint("POST", "/match", investor_profile, "Investor Matching")
    test_results.append(("Investor Matching", success))
    
    # ===================== TEST 3: RISK PREDICTION =====================
    # Test with minimal required data (based on api_examples.md)
    risk_prediction_minimal = {
        "nama_proyek": "Test Project Dashboard",
        "id_sektor": 3,
        "id_status": 3,
        "durasi_konsesi_tahun": 25,
        "nilai_investasi_total_idr": 5000000000000  # 5 trillion IDR
    }
    
    success, result = test_endpoint("POST", "/predict-risk", risk_prediction_minimal, "Risk Prediction (Minimal Data)")
    test_results.append(("Risk Prediction - Minimal", success))
    
    # Test with complete data
    risk_prediction_complete = {
        "nama_proyek": "Test Project Complete",
        "id_sektor": 16,
        "id_status": 6,
        "durasi_konsesi_tahun": 30,
        "nilai_investasi_total_idr": 8700000000000,
        "target_dana_tokenisasi_idr": 3000000000000,
        "persentase_tokenisasi": 34.5,
        "jenis_token_utama": "Energi",
        "token_risk_level_ordinal": 2,
        "token_ada_jaminan_pokok": True,
        "token_return_berbasis_kinerja": True,
        "dok_studi_kelayakan": True,
        "dok_laporan_keuangan_audit": True,
        "dok_peringkat_kredit": True
    }
    
    success, result = test_endpoint("POST", "/predict-risk", risk_prediction_complete, "Risk Prediction (Complete Data)")
    test_results.append(("Risk Prediction - Complete", success))
    
    # ===================== TEST 4: CHAT START =====================
    chat_start_payload = {
        "investor_profile": investor_profile,
        "user_name": "Test User Dashboard"
    }
    
    success, result = test_endpoint("POST", "/chat/start", chat_start_payload, "Chat Start")
    test_results.append(("Chat Start", success))
    
    # Store session ID for next test
    session_id = None
    if success and 'session_id' in result:
        session_id = result['session_id']
    
    # ===================== TEST 5: CHAT MESSAGE =====================
    if session_id:
        chat_message_payload = {
            "session_id": session_id,
            "message": "Saya tertarik dengan investasi infrastruktur. Bisa jelaskan keuntungannya?"
        }
        
        success, result = test_endpoint("POST", "/chat/message", chat_message_payload, "Chat Message")
        test_results.append(("Chat Message", success))
    else:
        print(f"\n⚠️ SKIPPING Chat Message test - no session_id from previous test")
        test_results.append(("Chat Message", False))
    
    # ===================== TEST 6: DATASET STATS =====================
    success, result = test_endpoint("GET", "/dataset/stats", description="Dataset Statistics")
    test_results.append(("Dataset Stats", success))
    
    # ===================== TEST 7: GEMINI TEST =====================
    success, result = test_endpoint("GET", "/test-gemini", description="Gemini API Test")
    test_results.append(("Gemini API", success))
    
    # ===================== SUMMARY =====================
    print(f"\n{'='*60}")
    print(f"📋 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in test_results if success)
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! API is working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Check the details above.")
        print(f"💡 Common issues:")
        print(f"   - API server not running: python api_matchmaker.py")
        print(f"   - Wrong endpoint URLs or parameters")
        print(f"   - Missing authentication token")
        print(f"   - Database/model files not found")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
