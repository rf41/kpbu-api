# KPBU Investor Matchmaker API

## 📋 Deskripsi
REST API untuk sistem matching investor dengan proyek KPBU (Kerjasama Pemerintah Badan Usaha) berdasarkan profil risiko dan preferensi investor.

## 🚀 Menjalankan di GitHub Codespace

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Menjalankan API
```bash
# Jalankan server
python api_matchmaker.py

# Atau menggunakan uvicorn langsung
uvicorn api_matchmaker:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Akses API
- **Local**: `http://localhost:8000`
- **Codespace**: Gunakan port forwarding untuk akses eksternal
- **Dokumentasi Interaktif**: `http://localhost:8000/docs`

## 🔐 Authentication
API menggunakan Bearer Token sederhana untuk prototyping:
```
Authorization: Bearer kpbu-matchmaker-2025
```

## 📡 Endpoints

### 1. Health Check
```http
GET /health
```

### 2. Get Available Sectors
```http
GET /sectors
Authorization: Bearer kpbu-matchmaker-2025
```

### 3. Match Investor
```http
POST /match
Authorization: Bearer kpbu-matchmaker-2025
Content-Type: application/json

{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [3, 1],
    "horison_investasi": "Jangka Menengah",
    "ukuran_investasi": "Menengah",
    "limit": 5
}
```

### 4. Predict Project Risk (NEW!)
```http
POST /predict-risk
Authorization: Bearer kpbu-matchmaker-2025
Content-Type: application/json

{
    "nama_proyek": "Jalan Tol Jakarta-Bandung",
    "id_sektor": 3,
    "id_status": 3,
    "durasi_konsesi_tahun": 25,
    "nilai_investasi_total_idr": 5500000000000,
    "jenis_token_utama": "Infrastruktur"
}
```

## 📝 Request Format

### InvestorProfile
```json
{
    "toleransi_risiko": "string",      // "Konservatif" | "Moderat" | "Agresif"
    "preferensi_sektor": [1, 3, 5],   // Array ID sektor yang diminati (1-16)
    "horison_investasi": "string",     // "Jangka Pendek" | "Jangka Menengah" | "Jangka Panjang"
    "ukuran_investasi": "string",      // "Kecil" | "Menengah" | "Besar" | "Semua"
    "limit": 5                         // Optional: jumlah rekomendasi (default: 5)
}
```

### ProjectData (NEW!)
```json
{
    "nama_proyek": "string",           // Nama proyek KPBU
    "id_sektor": 3,                    // ID sektor (1-16)
    "id_status": 3,                    // ID status proyek
    "durasi_konsesi_tahun": 25,        // Durasi konsesi dalam tahun
    "nilai_investasi_total_idr": 5500000000000,  // Nilai investasi dalam IDR
    "target_dana_tokenisasi_idr": 2000000000000, // Target dana tokenisasi (opsional)
    "persentase_tokenisasi": 36.4,     // Persentase tokenisasi (opsional)
    "jenis_token_utama": "string",     // Jenis token utama (opsional)
    "token_risk_level_ordinal": 3,     // Risk level token 1-5 (opsional)
    "token_ada_jaminan_pokok": false,  // Ada jaminan pokok (opsional)
    "token_return_berbasis_kinerja": true, // Return berbasis kinerja (opsional)
    "dok_studi_kelayakan": true,       // Dokumen studi kelayakan (opsional)
    "dok_laporan_keuangan_audit": false, // Dokumen laporan audit (opsional)
    "dok_peringkat_kredit": false      // Dokumen peringkat kredit (opsional)
}
```

## 📤 Response Format

```json
{
    "success": true,
    "message": "Matching completed successfully",
    "total_projects_analyzed": 150,
    "projects_after_filter": 25,
    "recommendations": [
        {
            "ranking": 1,
            "nama_proyek": "Jalan Tol ABC",
            "sektor": "Jalan dan Jembatan",
            "profil_risiko": "Menengah",
            "durasi_tahun": 20,
            "nilai_investasi_triliun": 5.5,
            "skor_kecocokan_persen": 85.2,
            "analisis_kecocokan": {
                "sektor_match": true,
                "horison_match": true,
                "risiko_match": true
            }
        }
    ],
    "statistics": {
        "highest_score": 85.2,
        "average_score": 72.1,
        "lowest_score": 65.3
    }
}
```

## 🧪 Testing dengan curl

### Get Sectors
```bash
curl -X GET "http://localhost:8000/sectors" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

### Match Investor
```bash
curl -X POST "http://localhost:8000/match" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [3, 1],
    "horison_investasi": "Jangka Menengah",
    "ukuran_investasi": "Menengah",
    "limit": 3
  }'
```

### Predict Project Risk (NEW!)
```bash
curl -X POST "http://localhost:8000/predict-risk" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "nama_proyek": "Jalan Tol Jakarta-Bandung",
    "id_sektor": 3,
    "id_status": 3,
    "durasi_konsesi_tahun": 25,
    "nilai_investasi_total_idr": 5500000000000,
    "jenis_token_utama": "Infrastruktur"
  }'
```

## 🔧 Konfigurasi Codespace

### 1. Port Forwarding
Setelah menjalankan API, buka tab "Ports" di Codespace dan forward port 8000:
- Visibility: Public (untuk akses eksternal)
- Port: 8000

### 2. Environment Variables (Optional)
Jika ingin menggunakan token dinamis:
```bash
export KPBU_AUTH_TOKEN="your-custom-token"
```

### 3. Auto-start (Optional)
Tambahkan ke `.devcontainer/devcontainer.json`:
```json
{
    "postCreateCommand": "pip install -r requirements.txt",
    "forwardPorts": [8000]
}
```

## 🐛 Troubleshooting

### Data File Not Found
Pastikan file `model/data/data_kpbu.csv` ada dan dapat diakses:
```bash
ls -la model/data/data_kpbu.csv
```

### Permission Errors
```bash
chmod +x api_matchmaker.py
```

### Port Already in Use
```bash
# Kill existing process
pkill -f "python api_matchmaker.py"

# Or use different port
uvicorn api_matchmaker:app --host 0.0.0.0 --port 8001
```

## 📊 Performance Notes
- API di-load dengan data CSV saat startup
- Untuk production, pertimbangkan database dan caching
- Memory usage bergantung pada ukuran dataset KPBU

## 🔄 Development Mode
Untuk development dengan auto-reload:
```bash
uvicorn api_matchmaker:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Setup Model Prediksi Risiko

Sebelum menggunakan endpoint `/predict-risk`, Anda perlu melatih model terlebih dahulu:

```bash
# Train dan simpan model prediksi risiko
python train_risk_model.py
```

Model akan disimpan di folder `model/saved_models/` dan dapat digunakan oleh API tanpa perlu training ulang.
