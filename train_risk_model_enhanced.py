"""
Script untuk melatih model prediksi risiko KPBU dengan Domain Knowledge Integration
Author: Ridwan Firmansyah
Description: Training model dengan integrasi bobot risiko domain knowledge untuk akurasi tinggi
Version: 3.0 - Domain Knowledge Enhanced with Risk Weights Integration
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

def load_domain_knowledge_weights():
    """Load domain knowledge risk weights dari JSON"""
    try:
        with open("risk_weights_optimized.json", "r", encoding="utf-8") as f:
            risk_weights = json.load(f)
        
        with open("risk_weights_lookup.json", "r", encoding="utf-8") as f:
            lookup_weights = json.load(f)
        
        print("✅ Domain knowledge risk weights loaded successfully")
        return risk_weights, lookup_weights
    except Exception as e:
        print(f"⚠️  Warning: Could not load domain knowledge weights: {e}")
        return None, None

def apply_domain_knowledge_features(df, lookup_weights):
    """Apply domain knowledge sebagai features baru"""
    
    if lookup_weights is None:
        print("⚠️  Creating basic risk features without domain knowledge weights")
        
        # Create basic risk features as fallback
        df['basic_sektor_risk'] = df['id_sektor'].map(lambda x: 0.5)  # Default medium risk
        df['basic_status_risk'] = df['id_status'].map(lambda x: 1.0)  # Default risk
        df['basic_duration_risk'] = df['Durasi_Konsesi_Tahun'].apply(
            lambda x: 1.2 if x < 10 else 0.7 if x <= 20 else 0.5 if x <= 30 else 0.9
        )
        df['basic_investment_risk'] = df['Nilai_Investasi_Total_IDR'].apply(
            lambda x: 0.9 if x < 50e9 else 0.5 if x < 1e12 else 0.6
        )
        
        # Basic token features
        df['basic_token_risk'] = df['Token_Risk_Level_Ordinal'] * 0.6
        df['basic_jaminan_risk'] = df['Token_Ada_Jaminan_Pokok'].apply(lambda x: 0 if x == 1 else 5.0)
        df['basic_doc_risk'] = df[['Dok_Studi_Kelayakan', 'Dok_Laporan_Keuangan_Audit']].sum(axis=1).apply(lambda x: 5.0 - x * 2.5)
        
        # Composite basic risk
        df['basic_composite_risk'] = (
            df['basic_sektor_risk'] + df['basic_status_risk'] + 
            df['basic_duration_risk'] + df['basic_investment_risk'] +
            df['basic_token_risk'] + df['basic_jaminan_risk'] + df['basic_doc_risk']
        )
        
        print(f"✅ Added 8 basic risk features")
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
        'Hak Pendapatan': 1.13  # Map ke Revenue-Sharing equivalent
    }
    
    df['dk_token_type_risk_score'] = df['Jenis_Token_Utama'].map(
        lambda x: token_type_mapping.get(x, 1.5)  # Default medium risk
    )
    
    # 6. Token Risk Level Score dari ordinal
    risk_level_scores = {1: 0.6, 2: 1.2, 3: 1.8, 4: 2.4, 5: 3.0}
    df['dk_token_risk_level_score'] = df['Token_Risk_Level_Ordinal'].map(
        lambda x: risk_level_scores.get(x, 1.8)
    )
    
    # 7. Characteristics Risk Scores
    df['dk_jaminan_pokok_risk'] = df['Token_Ada_Jaminan_Pokok'].apply(
        lambda x: 0 if x == 1 else 20.0  # Jika ada jaminan, risk berkurang
    )
    
    df['dk_return_kinerja_risk'] = df['Token_Return_Berbasis_Kinerja'].apply(
        lambda x: 10.0 if x == 1 else 0  # Return berbasis kinerja = lebih berisiko
    )
    
    df['dk_studi_kelayakan_risk'] = df['Dok_Studi_Kelayakan'].apply(
        lambda x: 0 if x else 10.0  # Tidak ada studi kelayakan = berisiko
    )
    
    df['dk_audit_keuangan_risk'] = df['Dok_Laporan_Keuangan_Audit'].apply(
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
    df['dk_investment_tokenization_interaction'] = df['dk_investment_risk_score'] * df['Persentase_Tokenisasi']
    df['dk_risk_level_documentation_interaction'] = df['dk_token_risk_level_score'] * (
        df['Dok_Studi_Kelayakan'].astype(int) + df['Dok_Laporan_Keuangan_Audit'].astype(int)
    )
    
    print(f"✅ Added {12} domain knowledge features")
    return df

def create_enhanced_features(df, lookup_weights):
    """Create enhanced features dengan domain knowledge"""
    
    # Apply domain knowledge features
    df = apply_domain_knowledge_features(df, lookup_weights)
    
    # Advanced financial ratios
    df['tokenization_ratio'] = df['Target_Dana_Tokenisasi_IDR'] / df['Nilai_Investasi_Total_IDR']
    df['investment_log'] = np.log1p(df['Nilai_Investasi_Total_IDR'])
    df['tokenization_log'] = np.log1p(df['Target_Dana_Tokenisasi_IDR'])
    
    # Risk concentration features (dengan fallback)
    if 'dk_composite_risk_score' in df.columns:
        df['high_risk_concentration'] = (
            (df['Token_Risk_Level_Ordinal'] >= 4).astype(int) +
            (df['Persentase_Tokenisasi'] >= 0.5).astype(int) +
            (df['dk_composite_risk_score'] >= df['dk_composite_risk_score'].median()).astype(int)
        )
    else:
        # Fallback tanpa domain knowledge
        df['high_risk_concentration'] = (
            (df['Token_Risk_Level_Ordinal'] >= 4).astype(int) +
            (df['Persentase_Tokenisasi'] >= 0.5).astype(int)
        )
    
    # Documentation completeness score
    df['documentation_completeness'] = (
        df['Dok_Studi_Kelayakan'].astype(int) +
        df['Dok_Laporan_Keuangan_Audit'].astype(int) +
        df['Dok_Peringkat_Kredit'].astype(int)
    ) / 3.0
    
    # Project maturity indicator
    status_maturity = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.8, 6: 0.9, 7: 1.0}
    df['project_maturity'] = df['id_status'].map(status_maturity)
    
    feature_count = len([col for col in df.columns if col.startswith('dk_')])
    print(f"✅ Enhanced features created with {feature_count} domain knowledge features")
    return df

def train_domain_enhanced_model():
    """Train model dengan domain knowledge enhancement"""
    
    print("🚀 Starting DOMAIN KNOWLEDGE ENHANCED risk prediction model training...")
    print("📋 New Features: Domain Knowledge Integration, Risk Weights, Expert Rules")
    
    # Path files
    data_path = 'data/data_kpbu_with_token.csv'
    model_dir = 'model/saved_models'
    
    # Buat direktori model jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # 1. Load domain knowledge weights
        risk_weights, lookup_weights = load_domain_knowledge_weights()
        
        # 2. Load data training
        print("📊 Loading training data...")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} projects")
        
        # 3. Create enhanced features dengan domain knowledge
        df_enhanced = create_enhanced_features(df.copy(), lookup_weights)
        
        # 4. Prepare features dan target
        print("🔧 Preparing features with domain knowledge...")
        
        # Select features for training (exclude ID dan target)
        feature_columns = [col for col in df_enhanced.columns if col not in [
            'ID_Proyek', 'Nama_Proyek', 'Profil_Risiko'
        ]]
        
        X = df_enhanced[feature_columns]
        y = df_enhanced['Profil_Risiko']
        
        print(f"Features selected: {len(feature_columns)}")
        print(f"Domain knowledge features: {[col for col in feature_columns if col.startswith('dk_')]}")
        
        # 5. Handle categorical variables
        categorical_columns = ['Jenis_Token_Utama']
        for col in categorical_columns:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # 6. Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        risk_classes = label_encoder.classes_
        print(f"Risk classes: {list(risk_classes)}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # 7. Enhanced SMOTE-like augmentation dengan domain knowledge
        print("⚖️  Applying domain knowledge guided data augmentation...")
        X_balanced, y_balanced = apply_domain_guided_augmentation(X, y_encoded, label_encoder, lookup_weights)
        print(f"Dataset augmented: {len(X)} → {len(X_balanced)} samples")
        
        # 8. Feature selection dengan domain knowledge priority
        print("🎯 Applying domain knowledge guided feature selection...")
        X_selected, selected_features = apply_domain_guided_feature_selection(X_balanced, y_balanced)
        
        # 9. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # 10. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 11. Train ensemble model dengan domain knowledge weighting
        print("🤖 Training domain knowledge enhanced ensemble model...")
        ensemble_model = train_domain_enhanced_ensemble(X_train_scaled, y_train, risk_weights)
        
        # 12. Comprehensive evaluation
        print("📊 Evaluating domain enhanced model...")
        performance_metrics = evaluate_domain_enhanced_model(
            ensemble_model, X_test_scaled, y_test, risk_classes, lookup_weights
        )
        
        # 13. Save enhanced model components
        model_files = save_domain_enhanced_model(
            ensemble_model, scaler, label_encoder, selected_features, 
            lookup_weights, performance_metrics, model_dir
        )
        
        print("💾 Domain enhanced model saved successfully!")
        print(f"📁 Model files:")
        for name, path in model_files.items():
            if os.path.exists(path):
                print(f"   ✅ {name}: {path}")
        
        # 14. Print comprehensive performance report
        print("\n📈 DOMAIN ENHANCED MODEL PERFORMANCE:")
        print(f"   🎯 F1-weighted: {performance_metrics['f1_weighted']:.3f}")
        print(f"   🎯 Accuracy: {performance_metrics['accuracy']:.3f}")
        print(f"   🎯 Domain Knowledge Score: {performance_metrics['dk_score']:.3f}")
        print(f"   🎯 Composite Score: {performance_metrics['composite_score']:.3f}")
        print(f"   🏆 Rating: {performance_metrics['rating']}")
        
        print("\n" + classification_report(
            y_test, ensemble_model.predict(X_test_scaled), 
            target_names=risk_classes, 
            zero_division=0
        ))
        
        return model_files
        
    except Exception as e:
        print(f"❌ Error training domain enhanced model: {e}")
        raise e

def apply_domain_guided_augmentation(X, y_encoded, label_encoder, lookup_weights):
    """Apply domain knowledge guided data augmentation"""
    
    # Hitung target count berdasarkan domain knowledge importance
    class_counts = pd.Series(y_encoded).value_counts().sort_index()
    max_count = class_counts.max()
    
    # Tingkatkan sampling untuk kelas yang secara domain knowledge lebih penting
    target_counts = {}
    for class_idx in range(len(label_encoder.classes_)):
        class_name = label_encoder.classes_[class_idx]
        # Kelas dengan risiko tinggi mendapat lebih banyak augmentasi
        if class_name in ['Tinggi', 'Sangat Tinggi']:
            target_counts[class_idx] = max_count * 1.2  # 20% lebih banyak
        elif class_name in ['Menengah', 'Sedang']:
            target_counts[class_idx] = max_count
        else:
            target_counts[class_idx] = max_count * 0.8  # 20% lebih sedikit
    
    X_augmented = X.copy()
    y_augmented = y_encoded.copy()
    
    for class_idx in range(len(label_encoder.classes_)):
        class_count = np.sum(y_encoded == class_idx)
        target_count = int(target_counts[class_idx])
        augment_count = target_count - class_count
        
        if augment_count > 0:
            class_data = X[y_encoded == class_idx]
            
            if len(class_data) > 0:
                augmented_samples = []
                
                for i in range(augment_count):
                    base_idx = np.random.randint(0, len(class_data))
                    base_sample = class_data.iloc[base_idx].copy()
                    
                    # Domain knowledge guided noise
                    noise_factors = {}
                    for col in base_sample.index:
                        if col.startswith('dk_'):
                            noise_factors[col] = 0.02  # Lower noise for domain knowledge features
                        elif 'risk' in col.lower():
                            noise_factors[col] = 0.03  # Low noise for risk features
                        else:
                            noise_factors[col] = 0.05  # Standard noise
                    
                    # Apply noise
                    for col in base_sample.index:
                        if base_sample[col] != 0 and col in noise_factors:
                            noise = np.random.normal(0, noise_factors[col] * abs(base_sample[col]))
                            base_sample[col] += noise
                    
                    augmented_samples.append(base_sample)
                
                if augmented_samples:
                    augmented_df = pd.DataFrame(augmented_samples, columns=X.columns)
                    X_augmented = pd.concat([X_augmented, augmented_df], ignore_index=True)
                    y_augmented = np.concatenate([y_augmented, [class_idx] * len(augmented_samples)])
    
    return X_augmented, y_augmented

def apply_domain_guided_feature_selection(X_balanced, y_balanced, top_k=20):
    """Apply domain knowledge guided feature selection"""
    
    # Hitung korelasi untuk semua fitur
    feature_correlations = {}
    for col in X_balanced.columns:
        try:
            corr = np.corrcoef(X_balanced[col], y_balanced)[0, 1]
            if not np.isnan(corr):
                feature_correlations[col] = abs(corr)
        except:
            feature_correlations[col] = 0
    
    # Prioritas untuk domain knowledge features
    priority_features = []
    standard_features = []
    
    for feature, corr in feature_correlations.items():
        if feature.startswith('dk_'):
            priority_features.append((feature, corr * 1.5))  # Boost domain knowledge features
        else:
            standard_features.append((feature, corr))
    
    # Combine dan sort
    all_features = priority_features + standard_features
    sorted_features = sorted(all_features, key=lambda x: x[1], reverse=True)
    
    # Select top features
    n_features_to_select = min(top_k, len(sorted_features))
    selected_features = [feature for feature, _ in sorted_features[:n_features_to_select]]
    
    # Ensure at least 50% domain knowledge features
    dk_features = [f for f in selected_features if f.startswith('dk_')]
    if len(dk_features) < len(selected_features) * 0.5:
        # Add more domain knowledge features
        additional_dk = [f for f, _ in priority_features if f not in selected_features][:5]
        selected_features.extend(additional_dk)
    
    print(f"Selected features: {len(selected_features)}")
    print(f"Domain knowledge features: {len([f for f in selected_features if f.startswith('dk_')])}")
    
    return X_balanced[selected_features], selected_features

def train_domain_enhanced_ensemble(X_train_scaled, y_train, risk_weights):
    """Train ensemble model dengan domain knowledge"""
    
    # Calculate class weights berdasarkan domain knowledge
    class_weights = 'balanced'  # Start with balanced
    
    # Ensemble dengan multiple algorithms
    models = {
        'logistic': LogisticRegression(
            random_state=42, 
            class_weight=class_weights,
            max_iter=2000,
            C=1.0
        ),
        'random_forest': RandomForestClassifier(
            random_state=42,
            class_weight=class_weights,
            n_estimators=100,
            max_depth=10
        ),
        'gradient_boost': GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1
        )
    }
    
    # Train dan evaluate setiap model
    best_model = None
    best_score = 0
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
        avg_score = scores.mean()
        
        print(f"   {name}: {avg_score:.3f} ± {scores.std():.3f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
    
    # Train best model
    best_model.fit(X_train_scaled, y_train)
    print(f"   Best model selected with CV score: {best_score:.3f}")
    
    return best_model

def evaluate_domain_enhanced_model(model, X_test_scaled, y_test, risk_classes, lookup_weights):
    """Comprehensive evaluation dengan domain knowledge metrics"""
    
    y_pred = model.predict(X_test_scaled)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Domain knowledge score (berdasarkan konsistensi dengan expert knowledge)
    dk_score = calculate_domain_knowledge_consistency_score(y_test, y_pred, risk_classes)
    
    # Composite score dengan domain knowledge weighting
    composite_score = (
        0.3 * f1_weighted + 
        0.2 * accuracy + 
        0.2 * f1_macro + 
        0.3 * dk_score  # Higher weight untuk domain knowledge consistency
    )
    
    # Enhanced rating system
    if composite_score >= 0.92:
        rating = "9.7/10 - Excellent with Domain Knowledge"
    elif composite_score >= 0.88:
        rating = "9.2/10 - Very Good with Domain Knowledge"
    elif composite_score >= 0.84:
        rating = "8.7/10 - Good with Domain Knowledge"
    elif composite_score >= 0.80:
        rating = "8.2/10 - Acceptable with Domain Knowledge"
    else:
        rating = f"{composite_score*10:.1f}/10 - Needs Improvement"
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'dk_score': dk_score,
        'composite_score': composite_score,
        'rating': rating
    }

def calculate_domain_knowledge_consistency_score(y_test, y_pred, risk_classes):
    """Calculate konsistensi prediksi dengan domain knowledge"""
    
    # Create mapping risk level ke numeric
    risk_to_numeric = {
        'Sangat Rendah': 1, 'Rendah': 2, 'Menengah': 3, 'Sedang': 3,
        'Tinggi': 4, 'Sangat Tinggi': 5
    }
    
    # Convert ke numeric
    y_test_numeric = [risk_to_numeric.get(risk_classes[y], 3) for y in y_test]
    y_pred_numeric = [risk_to_numeric.get(risk_classes[y], 3) for y in y_pred]
    
    # Calculate consistency metrics
    exact_matches = sum(1 for i, j in zip(y_test_numeric, y_pred_numeric) if i == j)
    close_matches = sum(1 for i, j in zip(y_test_numeric, y_pred_numeric) if abs(i - j) <= 1)
    
    exact_accuracy = exact_matches / len(y_test)
    close_accuracy = close_matches / len(y_test)
    
    # Domain knowledge consistency score
    dk_score = 0.7 * exact_accuracy + 0.3 * close_accuracy
    
    return dk_score

def save_domain_enhanced_model(model, scaler, label_encoder, selected_features, 
                              lookup_weights, performance_metrics, model_dir):
    """Save domain enhanced model components"""
    
    model_files = {
        'model': f'{model_dir}/domain_enhanced_risk_model.joblib',
        'scaler': f'{model_dir}/domain_enhanced_scaler.joblib', 
        'label_encoder': f'{model_dir}/domain_enhanced_label_encoder.joblib',
        'features': f'{model_dir}/domain_enhanced_features.joblib',
        'domain_weights': f'{model_dir}/domain_risk_weights.joblib',
        'metadata': f'{model_dir}/domain_enhanced_metadata.joblib'
    }
    
    # Save components
    joblib.dump(model, model_files['model'])
    joblib.dump(scaler, model_files['scaler'])
    joblib.dump(label_encoder, model_files['label_encoder'])
    joblib.dump(selected_features, model_files['features'])
    joblib.dump(lookup_weights, model_files['domain_weights'])
    
    # Save enhanced metadata
    metadata = {
        'model_type': 'DomainEnhancedEnsemble',
        'version': '3.0',
        'performance_metrics': performance_metrics,
        'feature_count': len(selected_features),
        'domain_features': [f for f in selected_features if f.startswith('dk_')],
        'domain_knowledge_integration': True,
        'risk_weights_version': '1.0'
    }
    joblib.dump(metadata, model_files['metadata'])
    
    # Generate enhanced summary report
    summary_report = f"""
=== DOMAIN KNOWLEDGE ENHANCED MODEL TRAINING SUMMARY ===
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

FINAL MODEL PERFORMANCE:
- Model Type: Domain Enhanced Ensemble
- F1-weighted: {performance_metrics['f1_weighted']:.3f}
- Accuracy: {performance_metrics['accuracy']:.3f}
- Domain Knowledge Score: {performance_metrics['dk_score']:.3f}
- Composite Score: {performance_metrics['composite_score']:.3f}
- Rating: {performance_metrics['rating']}

DOMAIN KNOWLEDGE INTEGRATION:
1. ✅ Risk weights dari expert knowledge
2. ✅ Sektor, status, durasi, investasi risk scoring
3. ✅ Token type dan risk level mapping
4. ✅ Characteristics risk assessment
5. ✅ Composite risk score calculation
6. ✅ Domain-guided feature selection
7. ✅ Expert knowledge consistency scoring

FEATURES SELECTED: {len(selected_features)}
Domain Knowledge Features: {len([f for f in selected_features if f.startswith('dk_')])}

USAGE:
- Use domain_enhanced_* files for production
- Includes expert knowledge risk scoring
- Higher accuracy through domain integration
===================================================
"""
    
    report_path = f'{model_dir}/domain_enhanced_training_report.txt'
    with open(report_path, 'w') as f:
        f.write(summary_report)
    
    model_files['report'] = report_path
    return model_files

if __name__ == "__main__":
    train_domain_enhanced_model()
