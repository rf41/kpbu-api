"""
FastAPI REST API untuk Model Matchmaker Investor KPBU
Author: Ridwan Firmansyah
Description: API untuk matching investor dengan proyek KPBU berdasarkan profil risiko dan preferensi
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import pandas as pd
import numpy as np
import uvicorn
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================== KONFIGURASI =====================
AUTH_TOKEN = "kpbu-matchmaker-2025"  # Token statis untuk prototyping
DATA_PATH = "data/data_kpbu.csv"
MODEL_DIR = "model/saved_models"

# ===================== SECURITY =====================
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# ===================== MODELS =====================
class InvestorProfile(BaseModel):
    toleransi_risiko: str = Field(..., description="Konservatif/Moderat/Agresif")
    preferensi_sektor: List[int] = Field(..., description="List ID sektor yang diminati (1-16)")
    horison_investasi: str = Field(..., description="Jangka Pendek/Jangka Menengah/Jangka Panjang")
    ukuran_investasi: str = Field(..., description="Kecil/Menengah/Besar/Semua")
    limit: Optional[int] = Field(5, description="Jumlah rekomendasi maksimal")

class ProjectData(BaseModel):
    nama_proyek: str = Field(..., description="Nama proyek KPBU")
    id_sektor: int = Field(..., description="ID sektor (1-16)")
    id_status: int = Field(..., description="ID status proyek")
    durasi_konsesi_tahun: int = Field(..., description="Durasi konsesi dalam tahun")
    nilai_investasi_total_idr: float = Field(..., description="Nilai investasi total dalam IDR")
    target_dana_tokenisasi_idr: Optional[float] = Field(None, description="Target dana tokenisasi")
    persentase_tokenisasi: Optional[float] = Field(None, description="Persentase tokenisasi")
    jenis_token_utama: Optional[str] = Field(None, description="Jenis token utama")
    token_risk_level_ordinal: Optional[int] = Field(None, description="Risk level token (1-5)")
    token_ada_jaminan_pokok: Optional[bool] = Field(None, description="Ada jaminan pokok")
    token_return_berbasis_kinerja: Optional[bool] = Field(None, description="Return berbasis kinerja")
    dok_studi_kelayakan: Optional[bool] = Field(None, description="Dokumen studi kelayakan")
    dok_laporan_keuangan_audit: Optional[bool] = Field(None, description="Dokumen laporan audit")
    dok_peringkat_kredit: Optional[bool] = Field(None, description="Dokumen peringkat kredit")

class RiskPredictionResponse(BaseModel):
    success: bool
    message: str
    project_data: dict
    prediction: dict
    probabilities: dict
    risk_analysis: dict

class ProjectRecommendation(BaseModel):
    ranking: int
    nama_proyek: str
    sektor: str
    profil_risiko: str
    durasi_tahun: int
    nilai_investasi_triliun: float
    skor_kecocokan_persen: float
    analisis_kecocokan: dict

class MatchingResponse(BaseModel):
    success: bool
    message: str
    total_projects_analyzed: int
    projects_after_filter: int
    recommendations: List[ProjectRecommendation]
    statistics: dict

# ===================== DATA LOADER =====================
class DataLoader:
    def __init__(self):
        self.df_proyek = None
        self.sektor_mapping = {
            1: 'Air dan Sanitasi', 2: 'Infrastruktur', 3: 'Jalan dan Jembatan',
            4: 'Transportasi Laut', 5: 'Transportasi Udara', 6: 'Transportasi Darat',
            7: 'Telekomunikasi', 8: 'Transportasi', 9: 'Telekomunikasi dan Informatika',
            10: 'Pengelolaan Sampah', 11: 'Konservasi Energi', 12: 'Sumber Daya Air',
            13: 'Kesehatan', 14: 'Penelitian dan Pengembangan', 15: 'Perumahan',
            16: 'Energi Terbarukan'
        }
        self.load_data()
    
    def load_data(self):
        try:
            self.df_proyek = pd.read_csv(DATA_PATH)
            self.df_proyek['Sektor_Proyek'] = self.df_proyek['id_sektor'].map(self.sektor_mapping)
            self.df_proyek = self.df_proyek.dropna(subset=['Profil_Risiko', 'Sektor_Proyek'])
        except Exception as e:
            raise Exception(f"Failed to load data: {str(e)}")
    
    def get_available_sectors(self):
        return sorted(self.df_proyek['Sektor_Proyek'].unique().tolist())
    
    def get_sectors_mapping(self):
        return self.sektor_mapping
    
    def convert_sector_ids_to_names(self, sector_ids: List[int]) -> List[str]:
        """Konversi ID sektor ke nama sektor"""
        return [self.sektor_mapping.get(id_sektor) for id_sektor in sector_ids if id_sektor in self.sektor_mapping]

# ===================== RISK PREDICTION ENGINE =====================
class RiskPredictionEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.df_sektor = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model dari joblib files"""
        try:
            model_files = {
                'model': f'{MODEL_DIR}/risk_prediction_model.joblib',
                'scaler': f'{MODEL_DIR}/risk_prediction_scaler.joblib',
                'label_encoder': f'{MODEL_DIR}/risk_prediction_label_encoder.joblib',
                'feature_columns': f'{MODEL_DIR}/risk_prediction_features.joblib',
                'sector_reference': f'{MODEL_DIR}/sector_reference.joblib'
            }
            
            # Check if all files exist
            missing_files = [name for name, path in model_files.items() if not os.path.exists(path)]
            if missing_files:
                print(f"⚠️  Missing model files: {missing_files}")
                print("Please run 'python train_risk_model.py' first")
                return
            
            # Load model components
            self.model = joblib.load(model_files['model'])
            self.scaler = joblib.load(model_files['scaler'])
            self.label_encoder = joblib.load(model_files['label_encoder'])
            self.feature_columns = joblib.load(model_files['feature_columns'])
            
            try:
                self.df_sektor = joblib.load(model_files['sector_reference'])
            except:
                self.df_sektor = None
            
            self.is_loaded = True
            print("✅ Risk prediction model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading risk prediction model: {e}")
            self.is_loaded = False
    
    def preprocess_project_data(self, project_data: dict):
        """Preprocess data proyek baru untuk prediksi"""
        # Convert ke DataFrame
        df = pd.DataFrame([project_data])
        
        # Handle missing values dengan default values
        defaults = {
            'target_dana_tokenisasi_idr': 0,
            'persentase_tokenisasi': 0,
            'jenis_token_utama': 'Tidak Ditentukan',
            'token_risk_level_ordinal': 3,
            'token_ada_jaminan_pokok': False,
            'token_return_berbasis_kinerja': False,
            'dok_studi_kelayakan': False,
            'dok_laporan_keuangan_audit': False,
            'dok_peringkat_kredit': False
        }
        
        for col, default_val in defaults.items():
            if col not in df.columns or df[col].isna().any():
                df[col] = default_val
        
        # Konversi boolean ke int
        bool_columns = [
            'token_ada_jaminan_pokok', 'token_return_berbasis_kinerja',
            'dok_studi_kelayakan', 'dok_laporan_keuangan_audit', 'dok_peringkat_kredit'
        ]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Merge dengan referensi sektor jika tersedia
        if self.df_sektor is not None:
            df = df.merge(
                self.df_sektor[['id_sektor', 'risk_rank']], 
                on='id_sektor', 
                how='left'
            )
            
            # Buat fitur tambahan berbasis risiko sektor
            df['sektor_risk_score'] = 15 - df['risk_rank'].fillna(8)
            df['sektor_risk_category'] = pd.cut(
                df['risk_rank'].fillna(8), 
                bins=[0, 3, 7, 11, 15], 
                labels=['Sangat_Tinggi', 'Tinggi', 'Menengah', 'Rendah']
            )
        
        # Drop kolom yang tidak diperlukan
        X = df.drop(['nama_proyek'], axis=1, errors='ignore')
        
        # Handle kolom kategorikal
        categorical_columns = []
        possible_categorical = ['jenis_token_utama', 'sektor_risk_category']
        
        for col in possible_categorical:
            if col in X.columns:
                categorical_columns.append(col)
        
        if categorical_columns:
            # Konversi nama kolom ke format yang konsisten dengan training
            column_mapping = {
                'jenis_token_utama': 'Jenis_Token_Utama',
                'sektor_risk_category': 'sektor_risk_category'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in X.columns:
                    X = X.rename(columns={old_col: new_col})
            
            # Update categorical columns list
            categorical_columns = [column_mapping.get(col, col) for col in categorical_columns]
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Reindex untuk konsistensi dengan training data
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        
        return X
    
    def predict_risk(self, project_data: dict):
        """Prediksi risiko proyek baru"""
        if not self.is_loaded:
            raise Exception("Model belum dimuat. Jalankan train_risk_model.py terlebih dahulu.")
        
        try:
            # Preprocess data
            X_processed = self.preprocess_project_data(project_data)
            
            # Scale features
            X_scaled = self.scaler.transform(X_processed)
            
            # Prediksi
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Convert ke label asli
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # Buat dictionary probabilitas
            prob_dict = {}
            for i, label in enumerate(self.label_encoder.classes_):
                prob_dict[label] = round(probabilities[i] * 100, 2)
            
            # Analisis risiko sektor
            risk_analysis = {}
            if self.df_sektor is not None and 'id_sektor' in project_data:
                sector_info = self.df_sektor[self.df_sektor['id_sektor'] == project_data['id_sektor']]
                if not sector_info.empty:
                    risk_rank = sector_info.iloc[0]['risk_rank']
                    risk_analysis = {
                        'sector_risk_rank': int(risk_rank),
                        'sector_risk_level': (
                            'Sangat Tinggi' if risk_rank <= 3 else 
                            'Tinggi' if risk_rank <= 7 else 
                            'Menengah' if risk_rank <= 11 else 'Rendah'
                        )
                    }
            
            return {
                'predicted_risk': predicted_label,
                'confidence': round(max(probabilities) * 100, 2),
                'probabilities': prob_dict,
                'risk_analysis': risk_analysis
            }
            
        except Exception as e:
            raise Exception(f"Error in risk prediction: {str(e)}")

# ===================== MATCHING ENGINE =====================
class MatchingEngine:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.df = data_loader.df_proyek
    
    def filter_by_risk(self, profile: dict):
        toleransi = profile['toleransi_risiko']
        if toleransi == 'Konservatif':
            allowed_risks = ['Rendah']
        elif toleransi == 'Moderat':
            allowed_risks = ['Rendah', 'Menengah']
        else:  # Agresif
            return self.df
        return self.df[self.df['Profil_Risiko'].isin(allowed_risks)]
    
    def filter_by_investment_size(self, df_filtered, profile: dict):
        if profile['ukuran_investasi'] == 'Semua':
            return df_filtered
        
        nilai_investasi = df_filtered['Nilai_Investasi_Total_IDR'].dropna()
        median_investasi = nilai_investasi.median()
        
        if profile['ukuran_investasi'] == 'Kecil':
            return df_filtered[df_filtered['Nilai_Investasi_Total_IDR'] <= median_investasi / 2]
        elif profile['ukuran_investasi'] == 'Besar':
            return df_filtered[df_filtered['Nilai_Investasi_Total_IDR'] >= median_investasi * 2]
        else:  # Menengah
            return df_filtered[
                (df_filtered['Nilai_Investasi_Total_IDR'] >= median_investasi / 2) & 
                (df_filtered['Nilai_Investasi_Total_IDR'] <= median_investasi * 2)
            ]
    
    def calculate_match_score(self, project, profile):
        bobot = {'sektor': 0.4, 'horison': 0.3, 'risiko': 0.3}
        
        # Konversi ID sektor ke nama sektor untuk matching
        preferred_sector_names = self.data_loader.convert_sector_ids_to_names(profile['preferensi_sektor'])
        
        # Skor sektor
        skor_sektor = 1.0 if project['Sektor_Proyek'] in preferred_sector_names else 0.1
        
        # Skor horison
        durasi = project['Durasi_Konsesi_Tahun']
        horison = profile['horison_investasi']
        
        if horison == 'Jangka Pendek' and durasi <= 10:
            skor_horison = 1.0
        elif horison == 'Jangka Menengah' and 11 <= durasi <= 25:
            skor_horison = 1.0
        elif horison == 'Jangka Panjang' and durasi > 25:
            skor_horison = 1.0
        else:
            if horison == 'Jangka Pendek':
                skor_horison = max(0.2, 1 - abs(durasi - 5) * 0.05)
            elif horison == 'Jangka Menengah':
                skor_horison = max(0.2, 1 - abs(durasi - 18) * 0.03)
            else:
                skor_horison = max(0.2, 1 - abs(durasi - 30) * 0.02)
        
        # Skor risiko
        toleransi = profile['toleransi_risiko']
        risiko_proyek = project['Profil_Risiko']
        
        if toleransi == 'Konservatif' and risiko_proyek == 'Rendah':
            skor_risiko = 1.0
        elif toleransi == 'Moderat' and risiko_proyek in ['Rendah', 'Menengah']:
            skor_risiko = 0.9 if risiko_proyek == 'Menengah' else 0.8
        elif toleransi == 'Agresif':
            skor_risiko = 1.0 if risiko_proyek == 'Tinggi' else 0.7
        else:
            skor_risiko = 0.5
        
        return (bobot['sektor'] * skor_sektor + 
                bobot['horison'] * skor_horison + 
                bobot['risiko'] * skor_risiko)
    
    def generate_analysis(self, project, profile):
        analysis = {}
        
        # Konversi ID sektor ke nama sektor untuk analisis
        preferred_sector_names = self.data_loader.convert_sector_ids_to_names(profile['preferensi_sektor'])
        
        # Analisis sektor
        analysis['sektor_match'] = project['Sektor_Proyek'] in preferred_sector_names
        
        # Analisis horison
        durasi = project['Durasi_Konsesi_Tahun']
        horison = profile['horison_investasi']
        
        if horison == 'Jangka Pendek':
            analysis['horison_match'] = durasi <= 10
        elif horison == 'Jangka Menengah':
            analysis['horison_match'] = 11 <= durasi <= 25
        else:
            analysis['horison_match'] = durasi > 25
        
        # Analisis risiko
        toleransi = profile['toleransi_risiko']
        risiko_proyek = project['Profil_Risiko']
        
        if toleransi == 'Konservatif':
            analysis['risiko_match'] = risiko_proyek == 'Rendah'
        elif toleransi == 'Moderat':
            analysis['risiko_match'] = risiko_proyek in ['Rendah', 'Menengah']
        else:
            analysis['risiko_match'] = True
        
        return analysis
    
    def get_recommendations(self, profile_data: dict):
        # Filter berdasarkan risiko
        filtered_risk = self.filter_by_risk(profile_data)
        
        # Filter berdasarkan investasi
        filtered_final = self.filter_by_investment_size(filtered_risk, profile_data)
        
        if filtered_final.empty:
            return {
                'total_analyzed': len(self.df),
                'after_filter': 0,
                'recommendations': []
            }
        
        # Hitung skor kecocokan
        filtered_final = filtered_final.copy()
        filtered_final['skor_kecocokan'] = filtered_final.apply(
            lambda row: self.calculate_match_score(row, profile_data), axis=1
        )
        
        # Urutkan dan ambil top recommendations
        recommendations = filtered_final.sort_values('skor_kecocokan', ascending=False)
        limit = profile_data.get('limit', 5)
        top_recommendations = recommendations.head(limit)
        
        # Format hasil
        results = []
        for i, (_, project) in enumerate(top_recommendations.iterrows(), 1):
            analysis = self.generate_analysis(project, profile_data)
            
            results.append({
                'ranking': i,
                'nama_proyek': project['Nama_Proyek'],
                'sektor': project['Sektor_Proyek'],
                'profil_risiko': project['Profil_Risiko'],
                'durasi_tahun': int(project['Durasi_Konsesi_Tahun']),
                'nilai_investasi_triliun': round(project['Nilai_Investasi_Total_IDR'] / 1e12, 2),
                'skor_kecocokan_persen': round(project['skor_kecocokan'] * 100, 1),
                'analisis_kecocokan': analysis
            })
        
        return {
            'total_analyzed': len(self.df),
            'after_filter': len(filtered_final),
            'recommendations': results,
            'statistics': {
                'highest_score': round(recommendations['skor_kecocokan'].max() * 100, 1),
                'average_score': round(recommendations['skor_kecocokan'].mean() * 100, 1),
                'lowest_score': round(recommendations['skor_kecocokan'].min() * 100, 1)
            }
        }

# ===================== FASTAPI APP =====================
app = FastAPI(
    title="KPBU Investor Matchmaker API",
    description="API untuk matching investor dengan proyek KPBU",
    version="1.0.0"
)

# Initialize components
data_loader = DataLoader()
matching_engine = MatchingEngine(data_loader)
risk_prediction_engine = RiskPredictionEngine()

@app.get("/")
async def root():
    return {"message": "KPBU Investor Matchmaker API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "projects_loaded": len(data_loader.df_proyek),
        "risk_model_loaded": risk_prediction_engine.is_loaded
    }

@app.get("/sectors")
async def get_available_sectors(token: str = Depends(verify_token)):
    """Mendapatkan daftar sektor yang tersedia dengan ID dan nama"""
    sectors_mapping = data_loader.get_sectors_mapping()
    available_sectors = data_loader.get_available_sectors()
    
    # Format response dengan ID dan nama
    sectors_with_id = []
    for sector_id, sector_name in sectors_mapping.items():
        if sector_name in available_sectors:
            sectors_with_id.append({
                "id": sector_id,
                "nama": sector_name
            })
    
    return {
        "sectors": sorted(sectors_with_id, key=lambda x: x['id']),
        "mapping": sectors_mapping
    }

@app.post("/match", response_model=MatchingResponse)
async def match_investor(
    profile: InvestorProfile, 
    token: str = Depends(verify_token)
):
    """Mendapatkan rekomendasi proyek berdasarkan profil investor"""
    try:
        # Validasi input
        valid_risk_levels = ['Konservatif', 'Moderat', 'Agresif']
        valid_horizons = ['Jangka Pendek', 'Jangka Menengah', 'Jangka Panjang']
        valid_sizes = ['Kecil', 'Menengah', 'Besar', 'Semua']
        
        if profile.toleransi_risiko not in valid_risk_levels:
            raise HTTPException(400, f"Invalid toleransi_risiko. Must be one of: {valid_risk_levels}")
        
        if profile.horison_investasi not in valid_horizons:
            raise HTTPException(400, f"Invalid horison_investasi. Must be one of: {valid_horizons}")
        
        if profile.ukuran_investasi not in valid_sizes:
            raise HTTPException(400, f"Invalid ukuran_investasi. Must be one of: {valid_sizes}")
        
        # Validasi ID sektor
        valid_sector_ids = list(data_loader.get_sectors_mapping().keys())
        invalid_sector_ids = [s for s in profile.preferensi_sektor if s not in valid_sector_ids]
        if invalid_sector_ids:
            raise HTTPException(400, f"Invalid sector IDs: {invalid_sector_ids}. Valid IDs: {valid_sector_ids}")
        
        # Process matching
        profile_dict = profile.dict()
        results = matching_engine.get_recommendations(profile_dict)
        
        return MatchingResponse(
            success=True,
            message="Matching completed successfully",
            total_projects_analyzed=results['total_analyzed'],
            projects_after_filter=results['after_filter'],
            recommendations=[ProjectRecommendation(**rec) for rec in results['recommendations']],
            statistics=results.get('statistics', {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/predict-risk", response_model=RiskPredictionResponse)
async def predict_project_risk(
    project: ProjectData,
    token: str = Depends(verify_token)
):
    """Prediksi profil risiko untuk proyek KPBU baru"""
    try:
        # Validasi ID sektor
        valid_sector_ids = list(data_loader.get_sectors_mapping().keys())
        if project.id_sektor not in valid_sector_ids:
            raise HTTPException(400, f"Invalid sector ID: {project.id_sektor}. Valid IDs: {valid_sector_ids}")
        
        # Validasi nilai investasi
        if project.nilai_investasi_total_idr <= 0:
            raise HTTPException(400, "Nilai investasi harus lebih besar dari 0")
        
        # Validasi durasi konsesi
        if project.durasi_konsesi_tahun <= 0:
            raise HTTPException(400, "Durasi konsesi harus lebih besar dari 0 tahun")
        
        if not risk_prediction_engine.is_loaded:
            raise HTTPException(500, "Risk prediction model is not loaded. Please run 'python train_risk_model.py' first.")
        
        # Convert pydantic model ke dict
        project_dict = project.dict()
        
        # Prediksi risiko
        prediction_result = risk_prediction_engine.predict_risk(project_dict)
        
        # Format response
        return RiskPredictionResponse(
            success=True,
            message="Risk prediction completed successfully",
            project_data={
                "nama_proyek": project.nama_proyek,
                "sektor": data_loader.sektor_mapping.get(project.id_sektor, "Unknown"),
                "id_sektor": project.id_sektor,
                "durasi_konsesi_tahun": project.durasi_konsesi_tahun,
                "nilai_investasi_triliun": round(project.nilai_investasi_total_idr / 1e12, 2),
                "jenis_token_utama": project.jenis_token_utama
            },
            prediction={
                "profil_risiko": prediction_result['predicted_risk'],
                "confidence_percent": prediction_result['confidence']
            },
            probabilities=prediction_result['probabilities'],
            risk_analysis=prediction_result['risk_analysis']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
