# Test API Examples for KPBU Matchmaker

## Example 1: Konservatif Investor

### Request:
```json
{
    "toleransi_risiko": "Konservatif",
    "preferensi_sektor": [1, 13],
    "horison_investasi": "Jangka Pendek",
    "ukuran_investasi": "Kecil",
    "limit": 3
}
```

### Response:
```json
{
    "success": true,
    "message": "Matching completed successfully",
    "total_projects_analyzed": 57,
    "projects_after_filter": 8,
    "recommendations": [
        {
            "ranking": 1,
            "nama_proyek": "Pembangunan SPAM Umbulan",
            "sektor": "Air dan Sanitasi",
            "profil_risiko": "Rendah",
            "durasi_tahun": 25,
            "nilai_investasi_triliun": 1.55,
            "skor_kecocokan_persen": 85.2,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        },
        {
            "ranking": 2,
            "nama_proyek": "RS Pendidikan Universitas Airlangga",
            "sektor": "Kesehatan",
            "profil_risiko": "Rendah",
            "durasi_tahun": 30,
            "nilai_investasi_triliun": 0.32,
            "skor_kecocokan_persen": 82.7,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        }
    ],
    "statistics": {
        "highest_score": 85.2,
        "average_score": 75.4,
        "lowest_score": 68.1
    }
}
```

## Example 2: Moderat Investor

### Request:
```json
{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [3, 8, 2],
    "horison_investasi": "Jangka Menengah",
    "ukuran_investasi": "Menengah",
    "limit": 5
}
```

### Response:
```json
{
    "success": true,
    "message": "Matching completed successfully",
    "total_projects_analyzed": 57,
    "projects_after_filter": 25,
    "recommendations": [
        {
            "ranking": 1,
            "nama_proyek": "Jalan Tol Cisumdawu",
            "sektor": "Jalan dan Jembatan",
            "profil_risiko": "Menengah",
            "durasi_tahun": 40,
            "nilai_investasi_triliun": 14.75,
            "skor_kecocokan_persen": 91.3,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        },
        {
            "ranking": 2,
            "nama_proyek": "Jalan Tol Serang-Panimbang",
            "sektor": "Jalan dan Jembatan",
            "profil_risiko": "Menengah",
            "durasi_tahun": 40,
            "nilai_investasi_triliun": 11.59,
            "skor_kecocokan_persen": 88.7,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        },
        {
            "ranking": 3,
            "nama_proyek": "Terminal Penumpang Pelabuhan Patimban",
            "sektor": "Infrastruktur",
            "profil_risiko": "Rendah",
            "durasi_tahun": 30,
            "nilai_investasi_triliun": 0.48,
            "skor_kecocokan_persen": 76.4,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        }
    ],
    "statistics": {
        "highest_score": 91.3,
        "average_score": 72.8,
        "lowest_score": 45.2
    }
}
```

## Example 3: Agresif Investor

### Request:
```json
{
    "toleransi_risiko": "Agresif", 
    "preferensi_sektor": [16, 9, 14],
    "horison_investasi": "Jangka Panjang",
    "ukuran_investasi": "Besar",
    "limit": 10
}
```

### Response:
```json
{
    "success": true,
    "message": "Matching completed successfully",
    "total_projects_analyzed": 57,
    "projects_after_filter": 45,
    "recommendations": [
        {
            "ranking": 1,
            "nama_proyek": "PLTS Terapung Cirata",
            "sektor": "Energi Terbarukan",
            "profil_risiko": "Tinggi",
            "durasi_tahun": 25,
            "nilai_investasi_triliun": 14.0,
            "skor_kecocokan_persen": 95.8,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        },
        {
            "ranking": 2,
            "nama_proyek": "Broadband Learning Center",
            "sektor": "Telekomunikasi dan Informatika",
            "profil_risiko": "Tinggi",
            "durasi_tahun": 35,
            "nilai_investasi_triliun": 2.5,
            "skor_kecocokan_persen": 92.1,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": true,
                "risiko_match": true
            }
        },
        {
            "ranking": 3,
            "nama_proyek": "PLTS Likupang",
            "sektor": "Energi Terbarukan",
            "profil_risiko": "Menengah",
            "durasi_tahun": 30,
            "nilai_investasi_triliun": 1.8,
            "skor_kecocokan_persen": 89.4,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": true,
                "risiko_match": true
            }
        }
    ],
    "statistics": {
        "highest_score": 95.8,
        "average_score": 78.3,
        "lowest_score": 52.7
    }
}
```

## Example 4: Diversified Investor

### Request:
```json
{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [1, 3, 6, 12, 15],
    "horison_investasi": "Jangka Menengah", 
    "ukuran_investasi": "Semua",
    "limit": 7
}
```

### Response:
```json
{
    "success": true,
    "message": "Matching completed successfully",
    "total_projects_analyzed": 57,
    "projects_after_filter": 38,
    "recommendations": [
        {
            "ranking": 1,
            "nama_proyek": "Jalan Tol Medan-Binjai",
            "sektor": "Jalan dan Jembatan",
            "profil_risiko": "Menengah",
            "durasi_tahun": 40,
            "nilai_investasi_triliun": 4.52,
            "skor_kecocokan_persen": 87.6,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": false,
                "risiko_match": true
            }
        },
        {
            "ranking": 2,
            "nama_proyek": "Sistem Penyediaan Air Minum Umbulan",
            "sektor": "Air dan Sanitasi",
            "profil_risiko": "Rendah",
            "durasi_tahun": 25,
            "nilai_investasi_triliun": 1.55,
            "skor_kecocokan_persen": 84.3,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": true,
                "risiko_match": true
            }
        },
        {
            "ranking": 3,
            "nama_proyek": "LRT Jabodebek",
            "sektor": "Transportasi Darat",
            "profil_risiko": "Tinggi",
            "durasi_tahun": 25,
            "nilai_investasi_triliun": 28.87,
            "skor_kecocokan_persen": 82.1,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": true,
                "risiko_match": false
            }
        }
    ],
    "statistics": {
        "highest_score": 87.6,
        "average_score": 71.9,
        "lowest_score": 48.4
    }
}
```

## RISK PREDICTION EXAMPLES (NEW!)

### Example 1: Jalan Tol Project

#### Request:
```json
{
    "nama_proyek": "Jalan Tol Jakarta-Bandung",
    "id_sektor": 3,
    "id_status": 3,
    "durasi_konsesi_tahun": 25,
    "nilai_investasi_total_idr": 5500000000000,
    "target_dana_tokenisasi_idr": 2000000000000,
    "persentase_tokenisasi": 36.4,
    "jenis_token_utama": "Infrastruktur",
    "token_risk_level_ordinal": 3,
    "token_ada_jaminan_pokok": false,
    "token_return_berbasis_kinerja": true,
    "dok_studi_kelayakan": true,
    "dok_laporan_keuangan_audit": true,
    "dok_peringkat_kredit": false
}
```

#### Response:
```json
{
    "success": true,
    "message": "Risk prediction completed successfully",
    "project_data": {
        "nama_proyek": "Jalan Tol Jakarta-Bandung",
        "sektor": "Jalan dan Jembatan",
        "id_sektor": 3,
        "durasi_konsesi_tahun": 25,
        "nilai_investasi_triliun": 5.5,
        "jenis_token_utama": "Infrastruktur"
    },
    "prediction": {
        "profil_risiko": "Menengah",
        "confidence_percent": 78.4
    },
    "probabilities": {
        "Rendah": 15.2,
        "Menengah": 78.4,
        "Tinggi": 6.4
    },
    "risk_analysis": {
        "sector_risk_rank": 8,
        "sector_risk_level": "Menengah"
    }
}
```

### Example 2: Air & Sanitasi Project

#### Request:
```json
{
    "nama_proyek": "Pembangunan PDAM Regional Jawa Barat",
    "id_sektor": 1,
    "id_status": 2,
    "durasi_konsesi_tahun": 20,
    "nilai_investasi_total_idr": 2300000000000,
    "jenis_token_utama": "Utilitas"
}
```

#### Response:
```json
{
    "success": true,
    "message": "Risk prediction completed successfully",
    "project_data": {
        "nama_proyek": "Pembangunan PDAM Regional Jawa Barat",
        "sektor": "Air dan Sanitasi",
        "id_sektor": 1,
        "durasi_konsesi_tahun": 20,
        "nilai_investasi_triliun": 2.3,
        "jenis_token_utama": "Utilitas"
    },
    "prediction": {
        "profil_risiko": "Rendah",
        "confidence_percent": 82.7
    },
    "probabilities": {
        "Rendah": 82.7,
        "Menengah": 14.8,
        "Tinggi": 2.5
    },
    "risk_analysis": {
        "sector_risk_rank": 5,
        "sector_risk_level": "Tinggi"
    }
}
```

### Example 3: Energi Terbarukan Project

#### Request:
```json
{
    "nama_proyek": "Pembangkit Listrik Tenaga Surya Bali",
    "id_sektor": 16,
    "id_status": 6,
    "durasi_konsesi_tahun": 30,
    "nilai_investasi_total_idr": 8700000000000,
    "jenis_token_utama": "Energi",
    "token_risk_level_ordinal": 2,
    "token_ada_jaminan_pokok": true,
    "dok_studi_kelayakan": true,
    "dok_laporan_keuangan_audit": true,
    "dok_peringkat_kredit": true
}
```

#### Response:
```json
{
    "success": true,
    "message": "Risk prediction completed successfully",
    "project_data": {
        "nama_proyek": "Pembangkit Listrik Tenaga Surya Bali",
        "sektor": "Energi Terbarukan",
        "id_sektor": 16,
        "durasi_konsesi_tahun": 30,
        "nilai_investasi_triliun": 8.7,
        "jenis_token_utama": "Energi"
    },
    "prediction": {
        "profil_risiko": "Tinggi",
        "confidence_percent": 91.2
    },
    "probabilities": {
        "Rendah": 3.1,
        "Menengah": 5.7,
        "Tinggi": 91.2
    },
    "risk_analysis": {
        "sector_risk_rank": 12,
        "sector_risk_level": "Rendah"
    }
}
```

### Example 4: Kesehatan Project (Minimal Data)

#### Request:
```json
{
    "nama_proyek": "Rumah Sakit Umum Daerah Modern",
    "id_sektor": 13,
    "id_status": 3,
    "durasi_konsesi_tahun": 15,
    "nilai_investasi_total_idr": 1200000000000
}
```

#### Response:
```json
{
    "success": true,
    "message": "Risk prediction completed successfully",
    "project_data": {
        "nama_proyek": "Rumah Sakit Umum Daerah Modern",
        "sektor": "Kesehatan",
        "id_sektor": 13,
        "durasi_konsesi_tahun": 15,
        "nilai_investasi_triliun": 1.2,
        "jenis_token_utama": null
    },
    "prediction": {
        "profil_risiko": "Rendah",
        "confidence_percent": 68.9
    },
    "probabilities": {
        "Rendah": 68.9,
        "Menengah": 25.6,
        "Tinggi": 5.5
    },
    "risk_analysis": {
        "sector_risk_rank": 6,
        "sector_risk_level": "Menengah"
    }
}
```

## cURL Commands

### Example 1 - Konservatif
```bash
curl -X POST "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/match" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "toleransi_risiko": "Konservatif",
    "preferensi_sektor": [1, 13],
    "horison_investasi": "Jangka Pendek",
    "ukuran_investasi": "Kecil",
    "limit": 3
  }'
```

### Example 2 - Moderat
```bash
curl -X POST "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/match" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [3, 8, 2],
    "horison_investasi": "Jangka Menengah", 
    "ukuran_investasi": "Menengah",
    "limit": 5
  }'
```

### RISK PREDICTION (NEW!)

#### Example 1 - Jalan Tol (Full Data)
```bash
curl -X POST "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/predict-risk" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "nama_proyek": "Jalan Tol Jakarta-Bandung",
    "id_sektor": 3,
    "id_status": 3,
    "durasi_konsesi_tahun": 25,
    "nilai_investasi_total_idr": 5500000000000,
    "target_dana_tokenisasi_idr": 2000000000000,
    "persentase_tokenisasi": 36.4,
    "jenis_token_utama": "Infrastruktur",
    "token_risk_level_ordinal": 3,
    "token_ada_jaminan_pokok": false,
    "token_return_berbasis_kinerja": true,
    "dok_studi_kelayakan": true,
    "dok_laporan_keuangan_audit": true,
    "dok_peringkat_kredit": false
  }'
```

#### Example 2 - Minimal Data
```bash
curl -X POST "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/predict-risk" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "nama_proyek": "Rumah Sakit Umum Daerah Modern",
    "id_sektor": 13,
    "id_status": 3,
    "durasi_konsesi_tahun": 15,
    "nilai_investasi_total_idr": 1200000000000
  }'
```

### Get Available Sectors

#### Request:
```bash
curl -X GET "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/sectors" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

#### Response:
```json
{
    "sectors": [
        {"id": 1, "nama": "Air dan Sanitasi"},
        {"id": 2, "nama": "Infrastruktur"},
        {"id": 3, "nama": "Jalan dan Jembatan"},
        {"id": 4, "nama": "Transportasi Laut"},
        {"id": 5, "nama": "Transportasi Udara"},
        {"id": 6, "nama": "Transportasi Darat"},
        {"id": 7, "nama": "Telekomunikasi"},
        {"id": 8, "nama": "Transportasi"},
        {"id": 9, "nama": "Telekomunikasi dan Informatika"},
        {"id": 10, "nama": "Pengelolaan Sampah"},
        {"id": 11, "nama": "Konservasi Energi"},
        {"id": 12, "nama": "Sumber Daya Air"},
        {"id": 13, "nama": "Kesehatan"},
        {"id": 14, "nama": "Penelitian dan Pengembangan"},
        {"id": 15, "nama": "Perumahan"},
        {"id": 16, "nama": "Energi Terbarukan"}
    ],
    "mapping": {
        "1": "Air dan Sanitasi",
        "2": "Infrastruktur",
        "3": "Jalan dan Jembatan",
        "4": "Transportasi Laut",
        "5": "Transportasi Udara",
        "6": "Transportasi Darat",
        "7": "Telekomunikasi",
        "8": "Transportasi",
        "9": "Telekomunikasi dan Informatika",
        "10": "Pengelolaan Sampah",
        "11": "Konservasi Energi",
        "12": "Sumber Daya Air",
        "13": "Kesehatan",
        "14": "Penelitian dan Pengembangan",
        "15": "Perumahan",
        "16": "Energi Terbarukan"
    }
}
```

### Health Check

#### Request:
```bash
curl -X GET "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/health"
```

#### Response:
```json
{
    "status": "healthy",
    "projects_loaded": 57,
    "risk_model_loaded": true
}
```

## Mapping Sektor ID ke Nama
```
1: Air dan Sanitasi
2: Infrastruktur
3: Jalan dan Jembatan
4: Transportasi Laut
5: Transportasi Udara
6: Transportasi Darat
7: Telekomunikasi
8: Transportasi
9: Telekomunikasi dan Informatika
10: Pengelolaan Sampah
11: Konservasi Energi
12: Sumber Daya Air
13: Kesehatan
14: Penelitian dan Pengembangan
15: Perumahan
16: Energi Terbarukan
```

## ERROR RESPONSE EXAMPLES

### 1. Invalid Authentication Token
#### Request:
```bash
curl -X POST "http://localhost:8000/match" \
  -H "Authorization: Bearer invalid-token" \
  -H "Content-Type: application/json"
```

#### Response (401 Unauthorized):
```json
{
    "detail": "Invalid authentication token"
}
```

### 2. Invalid Risk Tolerance
#### Request:
```json
{
    "toleransi_risiko": "InvalidLevel",
    "preferensi_sektor": [1, 3],
    "horison_investasi": "Jangka Pendek",
    "ukuran_investasi": "Kecil",
    "limit": 5
}
```

#### Response (400 Bad Request):
```json
{
    "detail": "Invalid toleransi_risiko. Must be one of: ['Konservatif', 'Moderat', 'Agresif']"
}
```

### 3. Invalid Sector ID in Risk Prediction
#### Request:
```json
{
    "nama_proyek": "Test Project",
    "id_sektor": 99,
    "id_status": 1,
    "durasi_konsesi_tahun": 20,
    "nilai_investasi_total_idr": 1000000000000
}
```

#### Response (400 Bad Request):
```json
{
    "detail": "Invalid sector ID: 99. Valid IDs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]"
}
```

### 4. Missing Required Fields
#### Request:
```json
{
    "nama_proyek": "Test Project",
    "id_sektor": 1
}
```

#### Response (422 Validation Error):
```json
{
    "detail": [
        {
            "loc": ["body", "durasi_konsesi_tahun"],
            "msg": "field required",
            "type": "value_error.missing"
        },
        {
            "loc": ["body", "nilai_investasi_total_idr"],
            "msg": "field required",
            "type": "value_error.missing"
        }
    ]
}
```

### 5. Risk Model Not Loaded
#### Response (500 Internal Server Error):
```json
{
    "detail": "Risk prediction model is not loaded. Please run 'python train_risk_model.py' first."
}
```

### 6. No Projects Found After Filtering
#### Request:
```json
{
    "toleransi_risiko": "Konservatif",
    "preferensi_sektor": [99],
    "horison_investasi": "Jangka Pendek",
    "ukuran_investasi": "Kecil",
    "limit": 5
}
```

#### Response (200 OK - No Results):
```json
{
    "success": true,
    "message": "Matching completed successfully",
    "total_projects_analyzed": 57,
    "projects_after_filter": 0,
    "recommendations": [],
    "statistics": {}
}
```

## TESTING WORKFLOW

### 1. Complete Testing Sequence
```bash
# 1. Check API Health
curl -X GET "http://localhost:8000/health"

# 2. Get Available Sectors
curl -X GET "http://localhost:8000/sectors" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"

# 3. Test Risk Prediction
curl -X POST "http://localhost:8000/predict-risk" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "nama_proyek": "Test Infrastructure Project",
    "id_sektor": 3,
    "id_status": 1,
    "durasi_konsesi_tahun": 25,
    "nilai_investasi_total_idr": 15000000000000,
    "target_dana_tokenisasi_idr": 3000000000000,
    "persentase_tokenisasi": 20.0,
    "jenis_token_utama": "Infrastructure Bond",
    "token_risk_level_ordinal": 3,
    "token_ada_jaminan_pokok": true,
    "token_return_berbasis_kinerja": false,
    "dok_studi_kelayakan": true,
    "dok_laporan_keuangan_audit": true,
    "dok_peringkat_kredit": false
  }'

# 4. Test Investor Matching
curl -X POST "http://localhost:8000/match" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [3, 1, 6],
    "horison_investasi": "Jangka Panjang",
    "ukuran_investasi": "Besar",
    "limit": 5
  }'
```

### 2. Python Testing Script
```python
import requests
import json

API_BASE_URL = "http://localhost:8000"
TOKEN = "kpbu-matchmaker-2025"

def test_api_endpoints():
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Health Check
    print("1. Testing Health Check...")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Get Sectors
    print("\n2. Testing Get Sectors...")
    response = requests.get(f"{API_BASE_URL}/sectors", headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Total Sectors: {len(response.json()['sectors'])}")
    
    # Test 3: Risk Prediction
    print("\n3. Testing Risk Prediction...")
    project_data = {
        "nama_proyek": "API Test Project",
        "id_sektor": 1,
        "id_status": 1,
        "durasi_konsesi_tahun": 20,
        "nilai_investasi_total_idr": 5000000000000,
        "jenis_token_utama": "Test Token"
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict-risk",
        headers=headers,
        json=project_data
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Risk: {result['prediction']['profil_risiko']}")
        print(f"Confidence: {result['prediction']['confidence_percent']}%")
    
    # Test 4: Investor Matching
    print("\n4. Testing Investor Matching...")
    investor_profile = {
        "toleransi_risiko": "Moderat",
        "preferensi_sektor": [1, 3],
        "horison_investasi": "Jangka Menengah",
        "ukuran_investasi": "Menengah",
        "limit": 3
    }
    
    response = requests.post(
        f"{API_BASE_URL}/match",
        headers=headers,
        json=investor_profile
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total Analyzed: {result['total_projects_analyzed']}")
        print(f"Recommendations: {len(result['recommendations'])}")
        if result['recommendations']:
            print(f"Top Recommendation: {result['recommendations'][0]['nama_proyek']}")

if __name__ == "__main__":
    test_api_endpoints()
```

## RESPONSE FIELD DESCRIPTIONS

### Investor Matching Response Fields
- `success`: Boolean indicating if the request was successful
- `message`: Human-readable status message
- `total_projects_analyzed`: Total number of projects in the dataset
- `projects_after_filter`: Number of projects remaining after applying filters
- `recommendations`: Array of recommended projects
  - `ranking`: Project ranking (1 = best match)
  - `nama_proyek`: Project name
  - `sektor`: Sector name
  - `profil_risiko`: Risk profile (Rendah/Menengah/Tinggi)
  - `durasi_tahun`: Project duration in years
  - `nilai_investasi_triliun`: Investment value in trillions IDR
  - `skor_kecocokan_persen`: Match score percentage
  - `analisis_kecocokan`: Detailed match analysis
- `statistics`: Score statistics (highest, average, lowest)

### Risk Prediction Response Fields
- `success`: Boolean indicating if the prediction was successful
- `message`: Human-readable status message
- `project_data`: Processed project information
- `prediction`: Main prediction results
  - `profil_risiko`: Predicted risk profile
  - `confidence_percent`: Prediction confidence percentage
- `probabilities`: Probability distribution for all risk levels
- `risk_analysis`: Sector-based risk analysis (if available)
