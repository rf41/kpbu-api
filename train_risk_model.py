"""
Script untuk melatih dan menyimpan model prediksi risiko KPBU menggunakan joblib
Author: Ridwan Firmansyah
Description: Training model prediksi risiko dan menyimpan ke file untuk digunakan di API
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def train_and_save_risk_model():
    """Train model prediksi risiko dan simpan ke file joblib"""
    
    # Path files
    data_path = 'data/data_kpbu_with_token.csv'
    ref_sektor_path = 'data/ref_sektor.csv'
    model_dir = 'model/saved_models'
    
    # Buat direktori model jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # 1. Load data training
        df = pd.read_csv(data_path)
        print(f" Loaded {len(df)} projects")
        
        # 2. Load referensi sektor
        try:
            df_sektor = pd.read_csv(ref_sektor_path)
            print(f" Loaded sector reference: {len(df_sektor)} sectors")
            
            # Merge dengan referensi sektor
            df = df.merge(df_sektor[['id_sektor', 'risk_rank']], on='id_sektor', how='left')
            
            # Buat fitur tambahan berbasis risiko sektor
            df['sektor_risk_score'] = 15 - df['risk_rank'].fillna(8)
            df['sektor_risk_category'] = pd.cut(
                df['risk_rank'].fillna(8), 
                bins=[0, 3, 7, 11, 15], 
                labels=['Sangat_Tinggi', 'Tinggi', 'Menengah', 'Rendah']
            )
            print("✅ Added sector risk features")
            
        except Exception as e:
            print(f"Warning: Could not load sector reference: {e}")
            df_sektor = None
        
        # 3. Persiapkan fitur dan target
        print("🔧 Preprocessing data...")
        X = df.drop(['ID_Proyek', 'Nama_Proyek', 'Profil_Risiko'], axis=1, errors='ignore')
        y = df['Profil_Risiko']
        
        print(f"Features before preprocessing: {list(X.columns)}")
        
        # Handle kolom kategorikal
        categorical_columns = []
        possible_categorical = ['Jenis_Token_Utama', 'sektor_risk_category']
        
        for col in possible_categorical:
            if col in X.columns:
                categorical_columns.append(col)
        
        if categorical_columns:
            print(f"🏷️  Encoding categorical columns: {categorical_columns}")
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Simpan nama kolom untuk konsistensi
        feature_columns = X.columns.tolist()
        print(f"Final features: {len(feature_columns)} columns")
        
        # 4. Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        risk_classes = label_encoder.classes_
        print(f"Risk classes: {list(risk_classes)}")
        
        # 5. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # 6. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 7. Train model
        print("Training LogisticRegression model...")
        model = LogisticRegression(
            random_state=42, 
            max_iter=1000, 
            solver='liblinear'
        )
        model.fit(X_train_scaled, y_train)
        
        # 8. Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2%}")
        
        # 9. Save semua komponen model
        model_files = {
            'model': f'{model_dir}/risk_prediction_model.joblib',
            'scaler': f'{model_dir}/risk_prediction_scaler.joblib',
            'label_encoder': f'{model_dir}/risk_prediction_label_encoder.joblib',
            'feature_columns': f'{model_dir}/risk_prediction_features.joblib',
            'sector_reference': f'{model_dir}/sector_reference.joblib'
        }
        
        # Save model components
        joblib.dump(model, model_files['model'])
        joblib.dump(scaler, model_files['scaler'])
        joblib.dump(label_encoder, model_files['label_encoder'])
        joblib.dump(feature_columns, model_files['feature_columns'])
        
        if df_sektor is not None:
            joblib.dump(df_sektor, model_files['sector_reference'])
        
        print("Model saved successfully!")
        print(f"Model files:")
        for name, path in model_files.items():
            if os.path.exists(path):
                print(f"    {name}: {path}")
        
        # 10. Print classification report
        print("\nModel Performance:")
        print(classification_report(
            y_test, y_pred, 
            target_names=risk_classes, 
            zero_division=0
        ))
        
        return model_files
        
    except Exception as e:
        print(f"Error training model: {e}")
        raise e

if __name__ == "__main__":
    train_and_save_risk_model()
