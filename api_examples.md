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
    },
    "data_saved": true,
    "saved_at": "2025-07-13T10:30:45.123456"
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

## CHATBOT API EXAMPLES (NEW!)

### Example 1: Start Chat Session

#### Request:
```json
{
    "investor_profile": {
        "toleransi_risiko": "Moderat",
        "preferensi_sektor": [3, 1, 6],
        "horison_investasi": "Jangka Panjang",
        "ukuran_investasi": "Besar",
        "limit": 5
    },
    "user_name": "Budi Santoso"
}
```

#### Response:
```json
{
    "success": true,
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "Selamat datang, Pak Budi Santoso! Saya adalah konsultan investasi KPBU yang akan membantu Anda menemukan peluang investasi terbaik. Berdasarkan profil investasi Anda (Moderat, Jangka Panjang, Investasi Besar), saya telah menyiapkan beberapa proyek KPBU unggulan yang sangat cocok untuk Anda:\n\n1. **Jalan Tol Medan-Binjai** (Jalan dan Jembatan)\n   - Profil Risiko: Menengah (sesuai toleransi Anda)\n   - Durasi: 40 tahun (investasi jangka panjang)\n   - Nilai Investasi: Rp 4.52 triliun\n   - Tingkat Kecocokan: 87.6% - Sangat Tinggi!\n\n2. **Sistem Penyediaan Air Minum Umbulan** (Air dan Sanitasi)\n   - Profil Risiko: Rendah (aman untuk portofolio)\n   - Durasi: 25 tahun\n   - Nilai Investasi: Rp 1.55 triliun\n   - Tingkat Kecocokan: 84.3%\n\n3. **LRT Jabodebek** (Transportasi Darat)\n   - Profil Risiko: Tinggi (diversifikasi portofolio)\n   - Durasi: 25 tahun\n   - Nilai Investasi: Rp 28.87 triliun\n   - Tingkat Kecocokan: 82.1%\n\nProyek-proyek ini dipilih berdasarkan algoritma AI canggih yang menganalisis 57 proyek KPBU aktif. Dengan track record pemerintah yang kuat dan demand infrastruktur Indonesia yang terus tumbuh, ini adalah momentum tepat untuk berinvestasi.\n\nApakah Anda ingin mengetahui lebih detail tentang salah satu proyek di atas, atau ada pertanyaan umum tentang investasi KPBU yang ingin Anda tanyakan?",
    "recommendations": [
        {
            "ranking": 1,
            "nama_proyek": "Jalan Tol Medan-Binjai",
            "sektor": "Jalan dan Jembatan",
            "profil_risiko": "Menengah",
            "durasi_tahun": 40,
            "nilai_investasi_triliun": 4.52,
            "skor_kecocokan_persen": 87.6
        },
        {
            "ranking": 2,
            "nama_proyek": "Sistem Penyediaan Air Minum Umbulan",
            "sektor": "Air dan Sanitasi",
            "profil_risiko": "Rendah",
            "durasi_tahun": 25,
            "nilai_investasi_triliun": 1.55,
            "skor_kecocokan_persen": 84.3
        }
    ],
    "suggested_questions": [
        "Berapa minimum investasi yang diperlukan?",
        "Bagaimana proses investasi KPBU dimulai?",
        "Apa keunggulan investasi KPBU dibanding instrumen lain?"
    ]
}
```

### Example 2: Chat Message - Project Inquiry

#### Request:
```json
{
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "Saya tertarik dengan proyek Jalan Tol Medan-Binjai. Bisa dijelaskan lebih detail tentang potensi keuntungannya?"
}
```

#### Response:
```json
{
    "success": true,
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "Pilihan yang sangat tepat, Pak Budi! Proyek Jalan Tol Medan-Binjai adalah investasi infrastruktur premium dengan beberapa keunggulan luar biasa:\n\n**💰 POTENSI KEUNTUNGAN FINANSIAL:**\n- ROI proyeksi: 12-15% per tahun (lebih tinggi dari deposito/obligasi)\n- Revenue stream stabil dari tarif tol selama 40 tahun\n- Nilai aset infrastruktur cenderung appreciation mengikuti inflasi\n- Diversifikasi portofolio dengan aset riil yang tangible\n\n**📈 FUNDAMENTAL KUAT:**\n- Lokasi strategis: menghubungkan Medan-Binjai (zona ekonomi vital)\n- Traffic growth proyeksi 8-12% per tahun seiring pertumbuhan ekonomi Sumut\n- Dukungan penuh pemerintah dengan skema KPBU yang proven\n- Demand infrastruktur transportasi yang tidak akan surut\n\n**🛡️ MITIGASI RISIKO:**\n- Government guarantee untuk revenue minimum\n- Struktur pembayaran yang predictable\n- Aset fisik sebagai underlying collateral\n- Track record positif proyek tol KPBU di Indonesia\n\n**🚀 TIMING SEMPURNA:**\n- Fase konstruksi dimulai 2025 (early investor advantage)\n- Indonesia masuk fase pertumbuhan ekonomi post-pandemic\n- Program 'Tol Trans Sumatera' mendapat prioritas tinggi pemerintah\n\nDengan investasi Rp 4.52 triliun dan tingkat kecocokan 87.6% dengan profil Anda, ini bisa menjadi 'cornerstone investment' yang menguntungkan dalam jangka panjang.\n\nApakah Anda ingin saya jelaskan struktur investasi dan cara memulainya? Atau ada aspek lain yang ingin diketahui lebih dalam?",
    "suggested_questions": [
        "Bagaimana struktur pembayaran dan minimum investasi?",
        "Apa saja dokumen yang diperlukan untuk memulai investasi?",
        "Bisakah dibandingkan dengan investasi konvensional lainnya?"
    ]
}
```

### Example 3: Chat Message - General KPBU Question

#### Request:
```json
{
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "Apa bedanya investasi KPBU dengan investasi saham atau properti?"
}
```

#### Response:
```json
{
    "success": true,
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "message": "Pertanyaan yang sangat cerdas, Pak Budi! Mari saya berikan perbandingan komprehensif:\n\n**🏗️ INVESTASI KPBU vs INSTRUMEN LAIN:**\n\n**KPBU (Infrastruktur):**\n✅ Return: 12-15% annually (stable & predictable)\n✅ Risiko: Moderate dengan government backing\n✅ Likuiditas: Medium (ada secondary market)\n✅ Durasi: Long-term (20-40 tahun) - compound growth\n✅ Inflation hedge: Excellent (tarif adjustable)\n✅ Social impact: Tinggi (nation building)\n\n**Saham:**\n⚠️ Return: 8-20% annually (volatile)\n⚠️ Risiko: High (market dependent)\n✅ Likuiditas: High\n⚠️ Durasi: Flexible tapi perlu market timing\n❌ Inflation hedge: Variable\n\n**Properti:**\n⚠️ Return: 8-12% annually\n⚠️ Risiko: Medium (location dependent)\n❌ Likuiditas: Low\n✅ Durasi: Long-term\n✅ Inflation hedge: Good\n\n**🎯 MENGAPA KPBU UNGGUL UNTUK PROFIL ANDA:**\n- **Stability**: Government partnership mengurangi political risk\n- **Predictability**: Revenue model yang terukur (traffic-based)\n- **Growth Potential**: Indonesia butuh Rp 4.000 triliun infrastruktur hingga 2045\n- **ESG Compliance**: Investasi berkelanjutan yang trending\n- **Portfolio Diversification**: Asset class berbeda dari traditional investment\n\n**💡 GOLDEN OPPORTUNITY:**\nSaat ini adalah timing sempurna karena:\n- Pemerintah prioritaskan KPBU untuk closing infrastructure gap\n- Interest rate environment yang supportive\n- Economic recovery yang sustainable\n\nDengan profil moderat dan horizon jangka panjang Anda, KPBU adalah 'sweet spot' yang menggabungkan safety dengan growth potential.\n\nApakah Anda siap untuk mengambil langkah konkret? Saya bisa jelaskan proses onboarding yang sederhana untuk memulai investasi Anda!",
    "suggested_questions": [
        "Bagaimana cara memulai proses investasi KPBU?",
        "Berapa minimum investasi untuk proyek yang direkomendasikan?",
        "Apakah ada program khusus untuk investor institusi?"
    ]
}
```

### Example 4: Get Chat Session Info

#### Request:
```bash
curl -X GET "http://localhost:8000/chat/session/a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

#### Response:
```json
{
    "success": true,
    "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "investor_profile": {
        "toleransi_risiko": "Moderat",
        "preferensi_sektor": [3, 1, 6],
        "horison_investasi": "Jangka Panjang",
        "ukuran_investasi": "Besar",
        "limit": 5
    },
    "recommendations_count": 5,
    "created_at": "2025-07-13T10:30:00Z",
    "user_name": "Budi Santoso"
}
```

### Example 5: Cleanup Chat Sessions

#### Request:
```bash
curl -X DELETE "http://localhost:8000/chat/cleanup" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

#### Response:
```json
{
    "success": true,
    "message": "Cleaned up 3 expired chat sessions",
    "active_sessions": 5
}
```

## CHATBOT cURL Commands

### Start Chat Session
```bash
curl -X POST "http://localhost:8000/chat/start" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "investor_profile": {
        "toleransi_risiko": "Moderat",
        "preferensi_sektor": [3, 1, 6],
        "horison_investasi": "Jangka Panjang",
        "ukuran_investasi": "Besar",
        "limit": 5
    },
    "user_name": "Budi Santoso"
  }'
```

### Send Chat Message
```bash
curl -X POST "http://localhost:8000/chat/message" \
  -H "Authorization: Bearer kpbu-matchmaker-2025" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id-here",
    "message": "Saya tertarik dengan proyek infrastruktur. Bagaimana cara memulai investasi?"
  }'
```

## CHATBOT WORKFLOW INTEGRATION

### 1. Frontend Integration Flow
```javascript
// 1. User clicks "Start Investment Chat"
const startChat = async (investorProfile, userName) => {
    const response = await fetch('/chat/start', {
        method: 'POST',
        headers: {
            'Authorization': 'Bearer kpbu-matchmaker-2025',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            investor_profile: investorProfile,
            user_name: userName
        })
    });
    
    const data = await response.json();
    // Store session_id for subsequent messages
    sessionStorage.setItem('chatSessionId', data.session_id);
    
    // Display AI greeting and project recommendations
    displayMessage(data.message, 'ai');
    showRecommendations(data.recommendations);
    showSuggestedQuestions(data.suggested_questions);
};

// 2. User sends message
const sendMessage = async (message) => {
    const sessionId = sessionStorage.getItem('chatSessionId');
    
    const response = await fetch('/chat/message', {
        method: 'POST',
        headers: {
            'Authorization': 'Bearer kpbu-matchmaker-2025',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            session_id: sessionId,
            message: message
        })
    });
    
    const data = await response.json();
    displayMessage(data.message, 'ai');
    showSuggestedQuestions(data.suggested_questions);
};
```

### 2. Complete User Journey
1. **Landing Page** → User fills investor profile form
2. **Matching Results** → System shows recommended projects
3. **"Start Chat" Button** → Initiates AI consultation
4. **AI Greeting** → Welcomes user with personalized project list
5. **Interactive Q&A** → User asks about projects/KPBU
6. **Investment Guidance** → AI provides persuasive, data-driven advice
7. **Call-to-Action** → AI guides toward investment decision

### 3. Persuasive AI Features
- **Data-Driven Arguments**: Uses real project metrics and ROI projections
- **Risk Mitigation**: Addresses concerns with government backing evidence
- **Social Proof**: References successful KPBU projects and track records
- **Urgency Creation**: Highlights market timing and early investor advantages
- **Personalization**: Tailors advice to individual investor profile
- **Solution-Oriented**: Converts objections into opportunities

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
- `data_saved`: Boolean indicating if prediction was saved to dataset
- `saved_at`: ISO timestamp when prediction was saved to CSV

### Dataset Statistics Response Fields
- `success`: Boolean indicating if statistics were retrieved successfully
- `dataset_info`: Information about dataset growth
  - `original_projects`: Number of projects in original dataset
  - `new_predictions`: Number of new predictions saved
  - `total_data_points`: Total data points available
  - `growth_percentage`: Percentage growth from new predictions
- `sector_distribution`: Count of projects by sector
- `risk_distribution`: Count of projects by risk level
- `investment_statistics`: Investment value statistics
- `latest_prediction`: Information about the most recent prediction
- `data_sources`: Paths to data files and model directory

## ERROR HANDLING FOR CHATBOT

### 1. Invalid Session ID
#### Response (404 Not Found):
```json
{
    "detail": "Chat session not found"
}
```

### 2. Gemini API Error
#### Response (500 Internal Server Error):
```json
{
    "detail": "Error processing chat message: Gemini API timeout"
}
```

### 3. Missing Required Fields
#### Response (422 Validation Error):
```json
{
    "detail": [
        {
            "loc": ["body", "investor_profile"],
            "msg": "field required",
            "type": "value_error.missing"
        }
    ]
}
```

## Get Dataset Statistics (NEW!)

#### Request:
```bash
curl -X GET "http://localhost:8000/dataset/stats" \
  -H "Authorization: Bearer kpbu-matchmaker-2025"
```

#### Response:
```json
{
    "success": true,
    "dataset_info": {
        "original_projects": 57,
        "new_predictions": 15,
        "total_data_points": 72,
        "growth_percentage": 26.32
    },
    "sector_distribution": {
        "Jalan dan Jembatan": 12,
        "Air dan Sanitasi": 8,
        "Energi Terbarukan": 7,
        "Transportasi Darat": 6,
        "Kesehatan": 5,
        "Infrastruktur": 4,
        "Telekomunikasi dan Informatika": 4,
        "Transportasi": 3,
        "Sumber Daya Air": 3,
        "Transportasi Laut": 2,
        "Transportasi Udara": 2,
        "Perumahan": 1
    },
    "risk_distribution": {
        "Menengah": 23,
        "Rendah": 20,
        "Tinggi": 14
    },
    "investment_statistics": {
        "total_value_triliun": 485.67,
        "average_value_triliun": 8.52,
        "max_value_triliun": 28.87,
        "min_value_triliun": 0.32
    },
    "latest_prediction": {
        "nama_proyek": "Jalan Tol Jakarta-Bandung",
        "predicted_risk": "Menengah",
        "prediction_confidence": 78.4,
        "prediction_timestamp": "2025-07-13T10:30:45.123456"
    },
    "data_sources": {
        "original_dataset": "data/data_kpbu.csv",
        "predictions_dataset": "data/data_kpbu_with_predictions.csv",
        "prediction_model": "model/saved_models/"
    }
}
```
