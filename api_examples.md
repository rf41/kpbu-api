# Test API Examples for KPBU Matchmaker

## Example 1: Konservatif Investor
```json
{
    "toleransi_risiko": "Konservatif",
    "preferensi_sektor": [1, 13],
    "horison_investasi": "Jangka Pendek",
    "ukuran_investasi": "Kecil",
    "limit": 3
}
```

## Example 2: Moderat Investor  
```json
{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [3, 8, 2],
    "horison_investasi": "Jangka Menengah",
    "ukuran_investasi": "Menengah",
    "limit": 5
}
```

## Example 3: Agresif Investor
```json
{
    "toleransi_risiko": "Agresif", 
    "preferensi_sektor": [16, 9, 14],
    "horison_investasi": "Jangka Panjang",
    "ukuran_investasi": "Besar",
    "limit": 10
}
```

## Example 4: Diversified Investor
```json
{
    "toleransi_risiko": "Moderat",
    "preferensi_sektor": [1, 3, 6, 12, 15],
    "horison_investasi": "Jangka Menengah", 
    "ukuran_investasi": "Semua",
    "limit": 7
}
```

## RISK PREDICTION EXAMPLES (NEW!)

### Example 1: Jalan Tol Project
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

### Example 2: Air & Sanitasi Project
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

### Example 3: Energi Terbarukan Project
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

### Example 4: Kesehatan Project (Minimal Data)
```json
{
    "nama_proyek": "Rumah Sakit Umum Daerah Modern",
    "id_sektor": 13,
    "id_status": 3,
    "durasi_konsesi_tahun": 15,
    "nilai_investasi_total_idr": 1200000000000
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
```bash
curl -X GET "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/sectors" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

### Health Check
```bash
curl -X GET "https://turbo-parakeet-v6qqq6qq4q66hx5gq-8000.app.github.dev/health"
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
