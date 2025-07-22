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
import google.generativeai as genai
import json
from datetime import datetime
import uuid
import warnings
warnings.filterwarnings('ignore')

# ===================== KONFIGURASI =====================
AUTH_TOKEN = "kpbu-matchmaker-2025"  # Token statis untuk prototyping
DATA_PATH = "data/data_kpbu.csv"
MODEL_DIR = "model/saved_models"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAzRRl-ydj19ZSF9IJXILPlfNioAJp5mho")  # Set environment variable

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

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
    data_saved: bool = Field(default=False, description="Whether prediction was saved to dataset")
    saved_at: Optional[str] = Field(None, description="Timestamp when data was saved")

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

# ===================== CHATBOT MODELS =====================
class ChatStartRequest(BaseModel):
    investor_profile: InvestorProfile
    user_name: Optional[str] = Field(None, description="Nama user untuk personalisasi")

class ChatMessage(BaseModel):
    session_id: str = Field(..., description="ID sesi chat")
    message: str = Field(..., description="Pesan dari user")

class ChatResponse(BaseModel):
    success: bool
    session_id: str
    message: str
    recommendations: Optional[List[dict]] = Field(None, description="Rekomendasi proyek jika chat dimulai")
    suggested_questions: Optional[List[str]] = Field(None, description="Pertanyaan yang disarankan")

# ===================== DATA LOADER =====================
class DataLoader:
    def __init__(self):
        self.df_proyek = None
        self.sektor_mapping = {
            1: "Air minum",
            2: "Infrastruktur", 
            3: "Jalan",
            4: "Kesehatan",
            5: "Ketenagalistrikan",
            6: "Fasilitas pendidikan, penelitian dan pengembangan",
            7: "Telekomunikasi dan informatika",
            8: "Transportasi",
            9: "Bangunan negara antara lain gedung perkantoran, rumah negara dan sarana pendukung lainnya",
            10: "Pemasyarakatan",
            11: "Konservasi energi",
            12: "Sumber daya air dan irigasi",
            13: "Sistem pengelolaan air limbah terpusat",
            14: "Sistem pengelolaan air limbah setempat",
            15: "Sistem pengelolaan persampahan dan/atau limbah bahan berbahaya dan beracun",
            16: "Minyak dan gas bumi dan energi terbarukan termasuk bio energi",
            17: "Fasilitas perkotaan",
            18: "Kawasan",
            19: "Pariwisata",
            20: "Fasilitas sarana olahraga, kesenian dan budaya",
            21: "Perumahan rakyat"
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

# ===================== DOMAIN ENHANCED RISK PREDICTION ENGINE =====================
class RiskPredictionEngine:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.domain_risk_weights = None
        self.model_metadata = None
        self.df_sektor = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load domain enhanced pre-trained model dari joblib files"""
        try:
            # Prioritas untuk model domain enhanced terbaru
            model_files = {
                'model': f'{MODEL_DIR}/domain_enhanced_risk_model.joblib',
                'scaler': f'{MODEL_DIR}/domain_enhanced_scaler.joblib',
                'label_encoder': f'{MODEL_DIR}/domain_enhanced_label_encoder.joblib',
                'feature_columns': f'{MODEL_DIR}/domain_enhanced_features.joblib',
                'domain_risk_weights': f'{MODEL_DIR}/domain_risk_weights.joblib',
                'metadata': f'{MODEL_DIR}/domain_enhanced_metadata.joblib',
                'sector_reference': f'{MODEL_DIR}/sector_reference.joblib'
            }
            
            # Check if domain enhanced files exist (sector_reference is optional)
            required_files = ['model', 'scaler', 'label_encoder', 'feature_columns', 'domain_risk_weights']
            missing_required = [name for name in required_files if not os.path.exists(model_files[name])]
            
            if missing_required:
                print(f"⚠️  Missing domain enhanced model files: {missing_required}")
                print("Please run 'python train_risk_model_enhanced.py' to generate domain enhanced models")
                return
            
            # Load domain enhanced model components
            self.model = joblib.load(model_files['model'])
            self.scaler = joblib.load(model_files['scaler'])
            self.label_encoder = joblib.load(model_files['label_encoder'])
            self.feature_columns = joblib.load(model_files['feature_columns'])
            self.domain_risk_weights = joblib.load(model_files['domain_risk_weights'])
            
            try:
                self.model_metadata = joblib.load(model_files['metadata'])
            except:
                self.model_metadata = None
            
            try:
                self.df_sektor = joblib.load(model_files['sector_reference'])
            except:
                self.df_sektor = None
            
            self.is_loaded = True
            model_version = self.model_metadata.get('version', 'domain_enhanced_v3.0') if self.model_metadata else 'domain_enhanced_v3.0'
            print(f"✅ Domain enhanced risk prediction model loaded successfully (version: {model_version})")
            
        except Exception as e:
            print(f"❌ Error loading domain enhanced risk prediction model: {e}")
            self.is_loaded = False
    
    def apply_domain_knowledge_features(self, df, lookup_weights):
        """Apply domain knowledge features berdasarkan risk weights"""
        if lookup_weights is None:
            print("⚠️ Domain risk weights not loaded, using basic fallback features")
            # Basic fallback features
            df['basic_sektor_risk'] = df['id_sektor'].apply(lambda x: 0.5)
            df['basic_status_risk'] = df['id_status'].apply(lambda x: 1.0) 
            df['basic_duration_risk'] = df['Durasi_Konsesi_Tahun'].apply(
                lambda x: 0.7 if x <= 20 else 0.9 if x <= 30 else 1.1
            )
            return df
        
        print("🧠 Applying domain knowledge features...")
        
        # 1. Sektor Risk Score dari domain knowledge
        df['dk_sektor_risk_score'] = df['id_sektor'].map(
            lambda x: lookup_weights['sektor_weights'].get(str(x), {}).get('risk_score', 0.32)
        )
        
        # 2. Status Risk Score dari domain knowledge  
        df['dk_status_risk_score'] = df['id_status'].map(
            lambda x: lookup_weights['status_weights'].get(str(x), {}).get('risk_score', 1.0)
        )
        
        # 3. Duration Risk Score berdasarkan durasi konsesi
        def get_duration_risk_score(duration):
            if duration < 10:
                return 1.18  # Kurang dari 10 tahun
            elif duration <= 20:
                return 0.71  # 10-20 tahun
            elif duration <= 30:
                return 0.47  # 21-30 tahun
            elif duration <= 40:
                return 0.71  # 31-40 tahun
            else:
                return 0.94  # Lebih dari 40 tahun
        
        df['dk_duration_risk_score'] = df['Durasi_Konsesi_Tahun'].apply(get_duration_risk_score)
        
        # 4. Investment Risk Score berdasarkan nilai investasi
        def get_investment_risk_score(investment):
            if investment < 50e9:  # < 50 miliar
                return 0.88
            elif investment < 250e9:  # 50M - 250M
                return 0.53
            elif investment < 1e12:  # 250M - 1T
                return 0.35
            elif investment < 5e12:  # 1T - 5T
                return 0.53
            else:  # >= 5T
                return 0.71
        
        df['dk_investment_risk_score'] = df['Nilai_Investasi_Total_IDR'].apply(get_investment_risk_score)
        
        # 5. Token Type Risk Score
        token_type_mapping = {
            'Asset-Backed Token (ABT)': 0.56,
            'Revenue-Sharing Token (RST)': 1.13,
            'Profit-Sharing Token': 1.69,
            'Availability Payment Token': 1.13,
            'Hybrid Token (gabungan RST + ABT)': 1.69,
            'Utility Token': 2.81,
            'Utang': 0.56,  # Map ke ABT equivalent
            'Ekuitas': 1.69,  # Map ke Profit-Sharing equivalent
            'Hibrida': 1.69,  # Map ke Hybrid equivalent
            'Hak Pendapatan': 1.13,  # Map ke Revenue-Sharing equivalent
            'Tidak Ditentukan': 1.5  # Default medium risk
        }
        
        df['dk_token_type_risk_score'] = df['Jenis_Token_Utama'].map(
            lambda x: token_type_mapping.get(x, 1.5)  # Default medium risk
        )
        
        # 6. Token Risk Level Score dari ordinal
        risk_level_scores = {1: 0.6, 2: 1.2, 3: 1.8, 4: 2.4, 5: 3.0}
        df['dk_token_risk_level_score'] = df['token_risk_level_ordinal'].map(
            lambda x: risk_level_scores.get(x, 1.8)
        )
        
        # 7. Characteristics Risk Scores
        df['dk_jaminan_pokok_risk'] = df['token_ada_jaminan_pokok'].apply(
            lambda x: 0 if x == 1 else 20.0  # Jika ada jaminan, risk berkurang
        )
        
        df['dk_return_kinerja_risk'] = df['token_return_berbasis_kinerja'].apply(
            lambda x: 10.0 if x == 1 else 0  # Return berbasis kinerja = lebih berisiko
        )
        
        df['dk_studi_kelayakan_risk'] = df['dok_studi_kelayakan'].apply(
            lambda x: 0 if x else 10.0  # Tidak ada studi kelayakan = berisiko
        )
        
        df['dk_audit_keuangan_risk'] = df['dok_laporan_keuangan_audit'].apply(
            lambda x: 0 if x else 10.0  # Tidak ada audit = berisiko
        )
        
        # 8. Composite Domain Knowledge Risk Score
        df['dk_composite_risk_score'] = (
            df['dk_sektor_risk_score'] * 0.07 +
            df['dk_status_risk_score'] * 0.06 +
            df['dk_duration_risk_score'] * 0.04 +
            df['dk_investment_risk_score'] * 0.03 +
            df['dk_token_type_risk_score'] * 0.09 +
            df['dk_token_risk_level_score'] * 0.09 +
            (df['dk_jaminan_pokok_risk'] + df['dk_return_kinerja_risk'] + 
             df['dk_studi_kelayakan_risk'] + df['dk_audit_keuangan_risk']) * 0.5
        )
        
        # 9. Risk Interaction Features
        df['dk_sektor_status_interaction'] = df['dk_sektor_risk_score'] * df['dk_status_risk_score']
        df['dk_investment_tokenization_interaction'] = df['dk_investment_risk_score'] * df['persentase_tokenisasi']
        df['dk_risk_level_documentation_interaction'] = df['dk_token_risk_level_score'] * (
            df['dok_studi_kelayakan'].astype(int) + df['dok_laporan_keuangan_audit'].astype(int)
        )
        
        print(f"✅ Added 14 domain knowledge features")
        return df

    def apply_enhanced_feature_engineering(self, X_df):
        """Apply enhanced feature engineering dengan domain knowledge integration"""
        
        # Apply domain knowledge features first
        X_df = self.apply_domain_knowledge_features(X_df, self.domain_risk_weights)
        
        # Advanced financial ratios
        if 'target_dana_tokenisasi_idr' in X_df.columns and 'Nilai_Investasi_Total_IDR' in X_df.columns:
            X_df['tokenization_ratio'] = X_df['target_dana_tokenisasi_idr'] / X_df['Nilai_Investasi_Total_IDR']
            X_df['investment_log'] = np.log1p(X_df['Nilai_Investasi_Total_IDR'])
            X_df['tokenization_log'] = np.log1p(X_df['target_dana_tokenisasi_idr'])
        
        # Risk concentration features (dengan fallback)
        if 'dk_composite_risk_score' in X_df.columns:
            X_df['high_risk_concentration'] = (
                (X_df['token_risk_level_ordinal'] >= 4).astype(int) +
                (X_df['persentase_tokenisasi'] >= 0.5).astype(int) +
                (X_df['dk_composite_risk_score'] >= X_df['dk_composite_risk_score'].median()).astype(int)
            )
        else:
            # Fallback tanpa domain knowledge
            X_df['high_risk_concentration'] = (
                (X_df['token_risk_level_ordinal'] >= 4).astype(int) +
                (X_df['persentase_tokenisasi'] >= 0.5).astype(int)
            )
        
        # Documentation completeness score
        X_df['documentation_completeness'] = (
            X_df['dok_studi_kelayakan'].astype(int) +
            X_df['dok_laporan_keuangan_audit'].astype(int) +
            X_df['dok_peringkat_kredit'].astype(int)
        )
        
        return X_df

    def preprocess_project_data(self, project_data: dict):
        """Preprocess data proyek baru untuk prediksi dengan domain enhanced pipeline"""
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
        
        # Konversi nama kolom untuk konsistensi dengan training
        column_mapping = {
            'nilai_investasi_total_idr': 'Nilai_Investasi_Total_IDR',
            'jenis_token_utama': 'Jenis_Token_Utama',
            'durasi_konsesi_tahun': 'Durasi_Konsesi_Tahun'
        }
        
        df = df.rename(columns=column_mapping)
        
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
            
            # Buat fitur tambahan berbasis risiko sektor (SAMA dengan training)
            df['sektor_risk_score'] = 15 - df['risk_rank'].fillna(8)
            df['sektor_risk_category'] = pd.cut(
                df['risk_rank'].fillna(8), 
                bins=[0, 3, 7, 11, 15], 
                labels=['Sangat_Tinggi', 'Tinggi', 'Menengah', 'Rendah']
            )
        
        # Drop kolom yang tidak diperlukan SEBELUM apply domain knowledge
        X = df.drop(['nama_proyek'], axis=1, errors='ignore')
        
        # Apply enhanced feature engineering dengan domain knowledge SEBELUM get_dummies
        X = self.apply_enhanced_feature_engineering(X)
        
        # Handle kolom kategorikal SETELAH domain knowledge features dibuat
        categorical_columns = []
        possible_categorical = ['Jenis_Token_Utama', 'sektor_risk_category']
        
        for col in possible_categorical:
            if col in X.columns:
                categorical_columns.append(col)
        
        if categorical_columns:
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Reindex untuk konsistensi dengan training data
        X = X.reindex(columns=self.feature_columns, fill_value=0)
        
        return X
    
    def predict_risk(self, project_data: dict):
        """Prediksi risiko proyek baru dengan domain enhanced model"""
        if not self.is_loaded:
            raise Exception("Domain enhanced model belum dimuat. Jalankan train_risk_model_enhanced.py terlebih dahulu untuk generate domain enhanced models.")
        
        try:
            # Preprocess data dengan domain enhanced pipeline
            X_processed = self.preprocess_project_data(project_data)
            
            # Scale features
            X_scaled = self.scaler.transform(X_processed)
            
            # Prediksi dengan domain enhanced model
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Convert ke label asli
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            
            # Buat dictionary probabilitas
            prob_dict = {}
            for i, label in enumerate(self.label_encoder.classes_):
                prob_dict[label] = round(probabilities[i] * 100, 2)
            
            # Enhanced confidence calculation
            max_prob = max(probabilities)
            confidence = round(max_prob * 100, 2)
            
            # Domain knowledge enhanced risk analysis
            risk_analysis = {}
            if self.domain_risk_weights is not None and 'id_sektor' in project_data:
                sektor_weights = self.domain_risk_weights.get('sektor_weights', {})
                sektor_info = sektor_weights.get(str(project_data['id_sektor']), {})
                
                if sektor_info:
                    risk_analysis = {
                        'domain_sektor_risk_score': sektor_info.get('risk_score', 0.5),
                        'domain_sektor_risk_level': sektor_info.get('kategori_risiko', 'Menengah'),
                        'domain_knowledge_applied': True,
                        'features_enhanced': 14
                    }
                    
                    # Adjust confidence based on domain knowledge consistency
                    domain_score = sektor_info.get('risk_score', 0.5)
                    if (predicted_label == 'Tinggi' and domain_score > 0.7) or \
                       (predicted_label == 'Rendah' and domain_score < 0.3) or \
                       (predicted_label == 'Menengah' and 0.3 <= domain_score <= 0.7):
                        confidence = min(95, confidence * 1.1)  # Boost confidence when consistent
                    
            # Prediction quality indicator with domain knowledge
            entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
            prediction_quality = "High" if entropy < 0.6 else "Medium" if entropy < 1.0 else "Low"
            
            # Model version from metadata
            model_version = self.model_metadata.get('version', 'domain_enhanced_v3.0') if self.model_metadata else 'domain_enhanced_v3.0'
            
            return {
                'predicted_risk': predicted_label,
                'confidence': round(confidence, 2),
                'probabilities': prob_dict,
                'risk_analysis': risk_analysis,
                'prediction_quality': prediction_quality,
                'model_version': model_version,
                'features_used': len(self.feature_columns),
                'domain_knowledge_integrated': self.domain_risk_weights is not None
            }
            
        except Exception as e:
            raise Exception(f"Error in domain enhanced risk prediction: {str(e)}")
    
    def save_prediction_to_csv(self, project_dict: dict, prediction_result: dict):
        """Simpan hasil prediksi ke CSV dataset untuk continuous learning dengan domain enhanced model"""
        try:
            # Path file untuk menyimpan data baru
            new_data_path = "data/data_kpbu_with_domain_enhanced_predictions.csv"
            
            # Prepare data untuk disimpan dengan enhanced features
            save_data = {
                'nama_proyek': project_dict.get('nama_proyek'),
                'id_sektor': project_dict.get('id_sektor'),
                'id_status': project_dict.get('id_status'),
                'durasi_konsesi_tahun': project_dict.get('durasi_konsesi_tahun'),
                'nilai_investasi_total_idr': project_dict.get('nilai_investasi_total_idr'),
                'target_dana_tokenisasi_idr': project_dict.get('target_dana_tokenisasi_idr', 0),
                'persentase_tokenisasi': project_dict.get('persentase_tokenisasi', 0),
                'jenis_token_utama': project_dict.get('jenis_token_utama', 'Tidak Ditentukan'),
                'token_risk_level_ordinal': project_dict.get('token_risk_level_ordinal', 3),
                'token_ada_jaminan_pokok': project_dict.get('token_ada_jaminan_pokok', False),
                'token_return_berbasis_kinerja': project_dict.get('token_return_berbasis_kinerja', False),
                'dok_studi_kelayakan': project_dict.get('dok_studi_kelayakan', False),
                'dok_laporan_keuangan_audit': project_dict.get('dok_laporan_keuangan_audit', False),
                'dok_peringkat_kredit': project_dict.get('dok_peringkat_kredit', False),
                'predicted_risk': prediction_result['predicted_risk'],
                'prediction_confidence': prediction_result['confidence'],
                'prediction_quality': prediction_result.get('prediction_quality', 'Medium'),
                'model_version': prediction_result.get('model_version', 'domain_enhanced_v3.0'),
                'features_used': prediction_result.get('features_used', 0),
                'domain_knowledge_integrated': prediction_result.get('domain_knowledge_integrated', False),
                'domain_sektor_risk_score': prediction_result.get('risk_analysis', {}).get('domain_sektor_risk_score'),
                'domain_sektor_risk_level': prediction_result.get('risk_analysis', {}).get('domain_sektor_risk_level'),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            # Konversi ke DataFrame
            new_row = pd.DataFrame([save_data])
            
            # Check if file exists
            if os.path.exists(new_data_path):
                # Load existing data and append
                existing_df = pd.read_csv(new_data_path)
                combined_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                # Create new file
                combined_df = new_row
            
            # Save to CSV
            combined_df.to_csv(new_data_path, index=False)
            
            return True, datetime.now().isoformat()
            
        except Exception as e:
            print(f"Error saving domain enhanced prediction to CSV: {e}")
            return False, None

# ===================== CHATBOT ENGINE =====================
class ChatBotEngine:
    def __init__(self):
        self.sessions = {}  # Store chat sessions in memory
        self.sessions_file = "data/chat_sessions.json"  # Persistent storage
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        self.load_sessions_from_file()
        
    def start_chat_session(self, investor_profile: dict, user_name: str = None, recommendations: List[dict] = None):
        """Start a new chat session with matched projects"""
        session_id = str(uuid.uuid4())
        
        # Create initial context with project recommendations
        projects_info = ""
        if recommendations:
            projects_info = "\\n\\nProyek KPBU yang direkomendasikan untuk profil investasi Anda:\\n"
            for i, proj in enumerate(recommendations[:5], 1):
                projects_info += f"{i}. {proj['nama_proyek']} ({proj['sektor']})\\n"
                projects_info += f"   - Profil Risiko: {proj['profil_risiko']}\\n"
                projects_info += f"   - Durasi: {proj['durasi_tahun']} tahun\\n"
                projects_info += f"   - Nilai Investasi: Rp {proj['nilai_investasi_triliun']} triliun\\n"
                projects_info += f"   - Skor Kecocokan: {proj['skor_kecocokan_persen']}%\\n\\n"
        
        # Create system prompt
        system_prompt = f"""
Anda adalah Partisipro seorang Agent konsultan investasi KPBU (Kerjasama Pemerintah dan Badan Usaha) yang berpengalaman dan profesional. 

PROFIL INVESTOR:
- Toleransi Risiko: {investor_profile.get('toleransi_risiko', 'Tidak disebutkan')}
- Horison Investasi: {investor_profile.get('horison_investasi', 'Tidak disebutkan')}
- Ukuran Investasi: {investor_profile.get('ukuran_investasi', 'Tidak disebutkan')}

{projects_info}

TUGAS ANDA:
1. Berikan sapaan yang ramah dan profesional
2. Informasikan proyek-proyek yang telah direkomendasikan sesuai profil investor
3. Ajukan pertanyaan terkait minat investor terhadap proyek tersebut
4. Dalam setiap respons, SELALU gunakan pendekatan yang halus untuk mengarahkan investor agar tertarik berinvestasi
5. Berikan informasi yang akurat tentang potensi keuntungan dan manfaat investasi KPBU
6. Jika ditanya tentang risiko, berikan penjelasan yang seimbang namun tetap positif
7. Gunakan data dan fakta untuk meyakinkan investor
8. Berikan jawaban yang ringkas dan to the point dan jangan terlalu banyak basa-basi

GAYA KOMUNIKASI:
- Profesional
- Persuasif 
- Berikan contoh konkret dan data
- Fokus pada peluang dan potensi keuntungan
- Jangan terlalu agresif

Mulai percakapan dengan menyapa {user_name if user_name else 'calon investor'} dan memperkenalkan rekomendasi proyek.
"""
        
        # Initialize chat with system prompt
        chat = self.model.start_chat(history=[])
        
        # Store session
        self.sessions[session_id] = {
            'chat': chat,
            'investor_profile': investor_profile,
            'recommendations': recommendations or [],
            'created_at': datetime.now(),
            'user_name': user_name
        }
        
        # Save to persistent storage
        self.save_sessions_to_file()
        
        # Generate initial response
        try:
            initial_response = chat.send_message(system_prompt)
            return session_id, initial_response.text
        except Exception as e:
            print(f"⚠️ Gemini API error in start_chat_session: {e}")
            print(f"⚠️ API Key configured: {GEMINI_API_KEY != 'your-gemini-api-key-here'}")
            fallback_message = f"Selamat datang{', ' + user_name if user_name else ''}! Saya adalah konsultan investasi KPBU yang siap membantu Anda.\n\n"
            
            if recommendations:
                fallback_message += "Berdasarkan profil investasi Anda, berikut adalah rekomendasi proyek KPBU terbaik:\n\n"
                for i, proj in enumerate(recommendations[:3], 1):
                    fallback_message += f"{i}. **{proj['nama_proyek']}** ({proj['sektor']})\n"
                    fallback_message += f"   - Profil Risiko: {proj['profil_risiko']}\n"
                    fallback_message += f"   - Nilai Investasi: Rp {proj['nilai_investasi_triliun']} triliun\n"
                    fallback_message += f"   - Skor Kecocokan: {proj['skor_kecocokan_persen']}%\n\n"
                
                fallback_message += "Proyek-proyek ini dipilih khusus berdasarkan preferensi risiko dan sektor Anda. "
            
            fallback_message += "Apakah ada yang ingin Anda ketahui lebih detail tentang investasi KPBU atau proyek-proyek ini?"
            
            return session_id, fallback_message
    
    def send_message(self, session_id: str, message: str):
        """Send message to existing chat session"""
        if session_id not in self.sessions:
            raise HTTPException(404, "Chat session not found")
        
        session = self.sessions[session_id]
        
        # Restore chat if needed
        if session['chat'] is None:
            self.restore_chat_session(session_id)
        
        chat = session['chat']
        
        # Add context to user message with investment guidance
        enhanced_message = f"""
User message: {message}

CONTEXT: Ini adalah percakapan dengan investor KPBU. Selalu arahkan respons untuk:
1. Menjawab pertanyaan dengan informatif
2. Menyoroti keuntungan dan peluang investasi
3. Memberikan alasan mengapa investasi KPBU menguntungkan
4. Jika membahas risiko, berikan solusi dan mitigasi
5. Berikan call-to-action yang halus untuk melanjutkan investasi
6. Berikan jawaban yang ringkas dan to the point

Proyek yang direkomendasikan untuk investor ini: {[proj['nama_proyek'] for proj in session['recommendations'][:3]]}

Berikan respons yang persuasif namun profesional.
"""
        
        try:
            response = chat.send_message(enhanced_message)
            
            # Update session timestamp
            session['last_activity'] = datetime.now()
            self.save_sessions_to_file()
            
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini API error in send_message: {e}")
            print(f"⚠️ Session ID: {session_id}")
            
            # Enhanced fallback response based on message content
            message_lower = message.lower()
            recommendations = [proj['nama_proyek'] for proj in session['recommendations'][:3]]
            
            if any(word in message_lower for word in ['return', 'untung', 'keuntungan', 'profit']):
                fallback_response = f"Berdasarkan analisis kami, proyek-proyek yang direkomendasikan seperti {', '.join(recommendations)} memiliki potensi return yang menarik. Investasi KPBU umumnya memberikan return stabil 8-15% per tahun dengan dukungan pemerintah yang kuat. Apakah Anda ingin mengetahui detail perhitungan ROI untuk proyek tertentu?"
            elif any(word in message_lower for word in ['risiko', 'risk', 'bahaya']):
                fallback_response = f"Risiko investasi pada proyek {', '.join(recommendations)} telah diminimalisir melalui skema KPBU dengan jaminan pemerintah. Selain itu, diversifikasi portofolio dengan beberapa proyek dapat mengurangi risiko keseluruhan. Sistem monitoring yang ketat juga memastikan transparansi dan akuntabilitas proyek."
            elif any(word in message_lower for word in ['mulai', 'start', 'invest', 'cara']):
                fallback_response = f"Untuk memulai investasi di proyek seperti {', '.join(recommendations)}, Anda dapat memulai dengan minimum investasi yang bervariasi tergantung proyek. Proses dimulai dengan registrasi, due diligence, dan penandatanganan kontrak investasi. Tim kami siap memandu Anda step-by-step."
            else:
                fallback_response = f"Terima kasih atas pertanyaan Anda. Proyek-proyek yang kami rekomendasikan ({', '.join(recommendations)}) telah melalui seleksi ketat dan memiliki track record yang solid. Investasi KPBU menawarkan keseimbangan antara return yang kompetitif dan risiko yang terkelola. Ada aspek khusus yang ingin Anda ketahui lebih dalam?"
            
            # Update session timestamp even on fallback
            session['last_activity'] = datetime.now()
            self.save_sessions_to_file()
            
            return fallback_response
    
    def get_session_info(self, session_id: str):
        """Get session information"""
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old chat sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            # Check last activity or created_at
            last_activity = session.get('last_activity', session['created_at'])
            if (current_time - last_activity).total_seconds() > max_age_hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        # Save updated sessions
        if expired_sessions:
            self.save_sessions_to_file()
        
        return len(expired_sessions)
    
    def load_sessions_from_file(self):
        """Load chat sessions from persistent storage"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    sessions_data = json.load(f)
                
                # Convert string timestamps back to datetime objects
                for session_id, session_data in sessions_data.items():
                    session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
                    # Don't restore chat object - will be recreated when needed
                    session_data['chat'] = None
                    self.sessions[session_id] = session_data
                
                print(f"✅ Loaded {len(self.sessions)} chat sessions from storage")
            else:
                print("📝 No existing chat sessions file found, starting fresh")
                
        except Exception as e:
            print(f"⚠️ Error loading chat sessions: {e}")
            self.sessions = {}
    
    def save_sessions_to_file(self):
        """Save chat sessions to persistent storage"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
            
            # Prepare data for JSON serialization
            sessions_data = {}
            for session_id, session in self.sessions.items():
                # Don't save chat object (not JSON serializable)
                session_copy = session.copy()
                session_copy['created_at'] = session_copy['created_at'].isoformat()
                session_copy.pop('chat', None)  # Remove chat object
                sessions_data[session_id] = session_copy
            
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving chat sessions: {e}")
    
    def restore_chat_session(self, session_id: str):
        """Restore chat object for existing session"""
        if session_id in self.sessions and self.sessions[session_id]['chat'] is None:
            try:
                # Recreate chat session with context
                session_data = self.sessions[session_id]
                investor_profile = session_data['investor_profile']
                recommendations = session_data['recommendations']
                user_name = session_data.get('user_name')
                
                # Create system prompt (same as start_chat_session)
                projects_info = ""
                if recommendations:
                    projects_info = "\\n\\nProyek KPBU yang direkomendasikan untuk profil investasi Anda:\\n"
                    for i, proj in enumerate(recommendations[:5], 1):
                        projects_info += f"{i}. {proj['nama_proyek']} ({proj['sektor']})\\n"
                        projects_info += f"   - Profil Risiko: {proj['profil_risiko']}\\n"
                        projects_info += f"   - Durasi: {proj['durasi_tahun']} tahun\\n"
                        projects_info += f"   - Nilai Investasi: Rp {proj['nilai_investasi_triliun']} triliun\\n"
                        projects_info += f"   - Skor Kecocokan: {proj['skor_kecocokan_persen']}%\\n\\n"
                
                system_prompt = f"""
Anda adalah seorang konsultan investasi KPBU yang sedang melanjutkan percakapan dengan {user_name if user_name else 'investor'}. 

KONTEKS SEBELUMNYA:
- Sesi chat dimulai pada: {session_data['created_at'].strftime('%Y-%m-%d %H:%M')}
- Profil Investor: Toleransi Risiko: {investor_profile.get('toleransi_risiko')}, Horison: {investor_profile.get('horison_investasi')}, Ukuran: {investor_profile.get('ukuran_investasi')}

{projects_info}

Lanjutkan percakapan dengan profesional dan persuasif, selalu mengarahkan ke investasi KPBU.
"""
                
                # Initialize new chat with context
                chat = self.model.start_chat(history=[])
                self.sessions[session_id]['chat'] = chat
                
                # Send context to warm up the chat
                try:
                    chat.send_message(system_prompt)
                except:
                    pass  # Continue even if context setup fails
                
            except Exception as e:
                print(f"⚠️ Error restoring chat session {session_id}: {e}")

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
chatbot_engine = ChatBotEngine()

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
        
        # Simpan prediksi ke CSV
        saved, saved_at = risk_prediction_engine.save_prediction_to_csv(project_dict, prediction_result)
        
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
            risk_analysis=prediction_result['risk_analysis'],
            data_saved=saved,
            saved_at=saved_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Internal server error: {str(e)}")

@app.post("/chat/start", response_model=ChatResponse)
async def start_chat(
    request: ChatStartRequest,
    token: str = Depends(verify_token)
):
    """Mulai sesi chat dengan rekomendasi proyek berdasarkan profil investor"""
    try:
        # Validasi profil investor
        profile_dict = request.investor_profile.dict()
        
        # Validasi input yang sama seperti endpoint match
        valid_risk_levels = ['Konservatif', 'Moderat', 'Agresif']
        valid_horizons = ['Jangka Pendek', 'Jangka Menengah', 'Jangka Panjang']
        valid_sizes = ['Kecil', 'Menengah', 'Besar', 'Semua']
        
        if profile_dict['toleransi_risiko'] not in valid_risk_levels:
            raise HTTPException(400, f"Invalid toleransi_risiko. Must be one of: {valid_risk_levels}")
        
        if profile_dict['horison_investasi'] not in valid_horizons:
            raise HTTPException(400, f"Invalid horison_investasi. Must be one of: {valid_horizons}")
        
        if profile_dict['ukuran_investasi'] not in valid_sizes:
            raise HTTPException(400, f"Invalid ukuran_investasi. Must be one of: {valid_sizes}")
        
        # Dapatkan rekomendasi proyek
        results = matching_engine.get_recommendations(profile_dict)
        recommendations = results['recommendations']
        
        # Mulai chat session dengan rekomendasi
        session_id, initial_message = chatbot_engine.start_chat_session(
            investor_profile=profile_dict,
            user_name=request.user_name,
            recommendations=recommendations
        )
        
        # Suggested questions
        suggested_questions = [
            f"Ceritakan lebih detail tentang proyek {recommendations[0]['nama_proyek'] if recommendations else 'yang direkomendasikan'}",
            "Berapa estimasi return yang bisa saya dapatkan?",
            "Bagaimana cara memulai investasi di proyek KPBU?",
            "Apa saja risiko investasi KPBU dan bagaimana mitigasinya?",
            "Apakah ada opsi investasi dengan nilai lebih kecil?"
        ]
        
        return ChatResponse(
            success=True,
            session_id=session_id,
            message=initial_message,
            recommendations=recommendations[:3] if recommendations else None,
            suggested_questions=suggested_questions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error starting chat session: {str(e)}")

@app.post("/chat/message", response_model=ChatResponse)
async def send_chat_message(
    request: ChatMessage,
    token: str = Depends(verify_token)
):
    """Kirim pesan ke sesi chat yang aktif"""
    try:
        # Kirim pesan ke chatbot
        response_message = chatbot_engine.send_message(request.session_id, request.message)
        
        # Get session info untuk context
        session_info = chatbot_engine.get_session_info(request.session_id)
        
        # Generate suggested follow-up questions based on context
        suggested_questions = []
        message_lower = request.message.lower()
        
        if any(word in message_lower for word in ['return', 'untung', 'keuntungan', 'profit']):
            suggested_questions = [
                "Bagaimana cara menghitung potensi ROI untuk proyek ini?",
                "Apakah ada historical return dari proyek KPBU serupa?",
                "Berapa lama waktu yang dibutuhkan untuk break even?"
            ]
        elif any(word in message_lower for word in ['risiko', 'risk', 'bahaya']):
            suggested_questions = [
                "Apa saja strategi mitigasi risiko yang tersedia?",
                "Bagaimana track record pemerintah dalam proyek KPBU?",
                "Apakah ada jaminan atau asuransi untuk investasi ini?"
            ]
        elif any(word in message_lower for word in ['mulai', 'start', 'invest', 'cara']):
            suggested_questions = [
                "Berapa minimum investasi yang diperlukan?",
                "Apa saja dokumen yang harus saya siapkan?",
                "Bagaimana proses due diligence dilakukan?"
            ]
        else:
            suggested_questions = [
                "Bisakah Anda jelaskan lebih detail tentang struktur investasi?",
                "Bagaimana perbandingan dengan investasi konvensional lainnya?",
                "Apakah ada proyek serupa yang sudah berhasil?"
            ]
        
        return ChatResponse(
            success=True,
            session_id=request.session_id,
            message=response_message,
            suggested_questions=suggested_questions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error processing chat message: {str(e)}")

@app.get("/chat/session/{session_id}")
async def get_chat_session(
    session_id: str,
    token: str = Depends(verify_token)
):
    """Dapatkan informasi sesi chat"""
    try:
        session_info = chatbot_engine.get_session_info(session_id)
        
        if not session_info:
            raise HTTPException(404, "Chat session not found")
        
        return {
            "success": True,
            "session_id": session_id,
            "investor_profile": session_info['investor_profile'],
            "recommendations_count": len(session_info['recommendations']),
            "created_at": session_info['created_at'].isoformat(),
            "last_activity": session_info.get('last_activity', session_info['created_at']).isoformat(),
            "user_name": session_info.get('user_name'),
            "is_active": session_info.get('chat') is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error retrieving chat session: {str(e)}")

@app.get("/chat/sessions")
async def get_all_chat_sessions(
    token: str = Depends(verify_token)
):
    """Dapatkan daftar semua sesi chat yang aktif"""
    try:
        sessions_info = []
        
        for session_id, session in chatbot_engine.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "user_name": session.get('user_name', 'Unknown'),
                "toleransi_risiko": session['investor_profile'].get('toleransi_risiko'),
                "created_at": session['created_at'].isoformat(),
                "last_activity": session.get('last_activity', session['created_at']).isoformat(),
                "recommendations_count": len(session['recommendations']),
                "is_active": session.get('chat') is not None
            })
        
        # Sort by last activity (most recent first)
        sessions_info.sort(key=lambda x: x['last_activity'], reverse=True)
        
        return {
            "success": True,
            "total_sessions": len(sessions_info),
            "sessions": sessions_info
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error retrieving chat sessions: {str(e)}")

@app.delete("/chat/cleanup")
async def cleanup_chat_sessions(
    token: str = Depends(verify_token)
):
    """Membersihkan sesi chat yang sudah kadaluarsa (lebih dari 24 jam)"""
    try:
        cleaned_count = chatbot_engine.cleanup_old_sessions()
        
        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} expired chat sessions",
            "active_sessions": len(chatbot_engine.sessions)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error cleaning up chat sessions: {str(e)}")

@app.get("/dataset/stats")
async def get_dataset_statistics(
    token: str = Depends(verify_token)
):
    """Mendapatkan statistik dataset termasuk prediksi yang tersimpan"""
    try:
        # Statistik dataset asli
        original_count = len(data_loader.df_proyek)
        
        # Check file prediksi baru dengan domain enhanced model
        new_data_path = "data/data_kpbu_with_domain_enhanced_predictions.csv"
        predictions_count = 0
        latest_prediction = None
        
        if os.path.exists(new_data_path):
            predictions_df = pd.read_csv(new_data_path)
            predictions_count = len(predictions_df)
            
            if predictions_count > 0:
                # Dapatkan prediksi terbaru
                latest_row = predictions_df.iloc[-1]
                latest_prediction = {
                    "nama_proyek": latest_row.get('nama_proyek'),
                    "predicted_risk": latest_row.get('predicted_risk'),
                    "prediction_confidence": latest_row.get('prediction_confidence'),
                    "prediction_timestamp": latest_row.get('prediction_timestamp')
                }
        
        # Statistik sektor
        sector_distribution = data_loader.df_proyek['Sektor_Proyek'].value_counts().to_dict()
        
        # Statistik profil risiko
        risk_distribution = data_loader.df_proyek['Profil_Risiko'].value_counts().to_dict()
        
        # Statistik nilai investasi
        investment_stats = {
            "total_value_triliun": round(data_loader.df_proyek['Nilai_Investasi_Total_IDR'].sum() / 1e12, 2),
            "average_value_triliun": round(data_loader.df_proyek['Nilai_Investasi_Total_IDR'].mean() / 1e12, 2),
            "max_value_triliun": round(data_loader.df_proyek['Nilai_Investasi_Total_IDR'].max() / 1e12, 2),
            "min_value_triliun": round(data_loader.df_proyek['Nilai_Investasi_Total_IDR'].min() / 1e12, 2)
        }
        
        return {
            "success": True,
            "dataset_info": {
                "original_projects": original_count,
                "new_predictions": predictions_count,
                "total_data_points": original_count + predictions_count,
                "growth_percentage": round((predictions_count / original_count) * 100, 2) if original_count > 0 else 0
            },
            "sector_distribution": sector_distribution,
            "risk_distribution": risk_distribution,
            "investment_statistics": investment_stats,
            "latest_prediction": latest_prediction,
            "data_sources": {
                "original_dataset": "data/data_kpbu.csv",
                "predictions_dataset": "data/data_kpbu_with_domain_enhanced_predictions.csv",
                "prediction_model": "model/saved_models/domain_enhanced_*",
                "domain_knowledge": "model/saved_models/domain_risk_weights.joblib"
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error retrieving dataset statistics: {str(e)}")

@app.get("/test-gemini")
async def test_gemini_connection(
    token: str = Depends(verify_token)
):
    """Test koneksi ke Gemini API"""
    try:
        # Test basic Gemini connection
        model = genai.GenerativeModel('gemini-1.5-flash')
        test_response = model.generate_content("Hello, test connection")
        
        return {
            "success": True,
            "message": "Gemini API connected successfully",
            "api_key_configured": GEMINI_API_KEY != "your-gemini-api-key-here",
            "api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY != "your-gemini-api-key-here" else 0,
            "test_response": test_response.text if test_response else "No response"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Gemini API connection failed: {str(e)}",
            "api_key_configured": GEMINI_API_KEY != "your-gemini-api-key-here",
            "api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY != "your-gemini-api-key-here" else 0,
            "error_type": type(e).__name__
        }
