"""
Script untuk melatih dan menyimpan model prediksi risiko KPBU (OPTIMIZED VERSION)
Author: Ridwan Firmansyah
Description: Training model prediksi risiko dengan optimasi lengkap untuk digunakan di API
Version: 2.0 - Optimized with SMOTE, Feature Engineering, and Hyperparameter Tuning
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

def train_and_save_risk_model():
    """Train optimized model prediksi risiko dan simpan ke file joblib"""
    
    print("🚀 Starting OPTIMIZED risk prediction model training...")
    print("📋 Improvements: SMOTE, Feature Engineering, Hyperparameter Tuning, Cross-Validation")
    
    # Path files
    data_path = 'data/data_kpbu_with_token.csv'
    ref_sektor_path = 'model/data/ref_sektor.csv'
    model_dir = 'model/saved_models'
    
    # Buat direktori model jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # 1. Load data training
        print("📊 Loading training data...")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} projects")
        
        # 2. Load referensi sektor dan buat fitur risiko sektor
        try:
            df_sektor = pd.read_csv(ref_sektor_path)
            print(f"✅ Loaded sector reference: {len(df_sektor)} sectors")
            
            # Merge dengan referensi sektor
            df = df.merge(df_sektor[['id_sektor', 'risk_rank']], on='id_sektor', how='left')
            
            # Buat fitur tambahan berbasis risiko sektor (SAMA dengan notebook)
            df['sektor_risk_score'] = 15 - df['risk_rank'].fillna(8)
            df['sektor_risk_category'] = pd.cut(
                df['risk_rank'].fillna(8), 
                bins=[0, 3, 7, 11, 15], 
                labels=['Sangat_Tinggi', 'Tinggi', 'Menengah', 'Rendah']
            )
            print("✅ Added sector risk features")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load sector reference: {e}")
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
        
        # 4. Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        risk_classes = label_encoder.classes_
        print(f"Risk classes: {list(risk_classes)}")
        
        # 5. IMPROVED CLASS BALANCING dengan SMOTE-like augmentation
        print("⚖️  Applying SMOTE-like data augmentation...")
        X_balanced, y_balanced = apply_smote_like_augmentation(X, y_encoded, label_encoder)
        print(f"Dataset augmented: {len(X)} → {len(X_balanced)} samples")
        
        # 6. ADVANCED FEATURE ENGINEERING
        print("🔬 Applying advanced feature engineering...")
        X_balanced_df = apply_advanced_feature_engineering(X_balanced, X.columns)
        
        # 7. FEATURE SELECTION
        print("🎯 Applying correlation-based feature selection...")
        X_balanced_selected = apply_feature_selection(X_balanced_df, y_balanced, top_k=15)
        feature_columns = X_balanced_selected.columns.tolist()
        print(f"Selected {len(feature_columns)} best features")
        
        # 8. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced_selected, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        
        # 9. Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 10. HYPERPARAMETER TUNING dengan GridSearchCV
        print("🤖 Training optimized LogisticRegression with GridSearchCV...")
        model = train_optimized_model(X_train_scaled, y_train)
        
        # 11. Comprehensive evaluation
        print("📊 Evaluating model performance...")
        performance_metrics = evaluate_model_comprehensive(model, X_test_scaled, y_test, risk_classes)
        
        # 12. Save optimized model components
        model_files = save_optimized_model(
            model, scaler, label_encoder, feature_columns, df_sektor, 
            performance_metrics, model_dir
        )
        
        print("💾 Optimized model saved successfully!")
        print(f"📁 Model files:")
        for name, path in model_files.items():
            if os.path.exists(path):
                print(f"   ✅ {name}: {path}")
        
        # 13. Print comprehensive performance report
        print("\n📈 OPTIMIZED MODEL PERFORMANCE:")
        print(f"   🎯 F1-weighted: {performance_metrics['f1_weighted']:.3f}")
        print(f"   🎯 Accuracy: {performance_metrics['accuracy']:.3f}")
        print(f"   🎯 Composite Score: {performance_metrics['composite_score']:.3f}")
        print(f"   🏆 Rating: {performance_metrics['rating']}")
        
        print("\n" + classification_report(
            y_test, model.predict(X_test_scaled), 
            target_names=risk_classes, 
            zero_division=0
        ))
        
        return model_files
        
    except Exception as e:
        print(f"❌ Error training optimized model: {e}")
        raise e


def apply_smote_like_augmentation(X, y_encoded, label_encoder):
    """Apply SMOTE-like data augmentation untuk class balancing"""
    
    # Hitung target count berdasarkan kelas mayoritas
    class_counts = pd.Series(y_encoded).value_counts().sort_index()
    max_count = class_counts.max()
    target_count = max_count
    
    X_augmented = X.copy()
    y_augmented = y_encoded.copy()
    
    # Augmentasi untuk setiap kelas yang kurang dari target
    for class_idx in range(len(label_encoder.classes_)):
        class_count = np.sum(y_encoded == class_idx)
        augment_count = target_count - class_count
        
        if augment_count > 0:
            # Ambil data dari kelas ini
            class_data = X[y_encoded == class_idx]
            
            if len(class_data) > 0:
                augmented_samples = []
                
                for i in range(augment_count):
                    # Pilih random sample dari kelas ini
                    base_idx = np.random.randint(0, len(class_data))
                    base_sample = class_data.iloc[base_idx].copy()
                    
                    # Variasi teknik augmentasi
                    augmentation_method = i % 3
                    
                    if augmentation_method == 0:
                        # Gaussian noise
                        noise_factor = np.random.uniform(0.03, 0.07)
                        for col in base_sample.index:
                            if base_sample[col] != 0:
                                noise = np.random.normal(0, noise_factor * abs(base_sample[col]))
                                base_sample[col] += noise
                                
                    elif augmentation_method == 1:
                        # Random perturbation
                        perturbation_factor = np.random.uniform(0.01, 0.05)
                        for col in base_sample.index:
                            if base_sample[col] != 0:
                                perturbation = base_sample[col] * np.random.uniform(-perturbation_factor, perturbation_factor)
                                base_sample[col] += perturbation
                                
                    else:
                        # Interpolation dengan sample lain
                        if len(class_data) > 1:
                            other_idx = np.random.randint(0, len(class_data))
                            while other_idx == base_idx:
                                other_idx = np.random.randint(0, len(class_data))
                            
                            other_sample = class_data.iloc[other_idx]
                            alpha = np.random.uniform(0.2, 0.8)
                            base_sample = alpha * base_sample + (1 - alpha) * other_sample
                    
                    # Ensure non-negative values for certain features
                    for col in base_sample.index:
                        if 'Token' in col or 'risk_score' in col:
                            base_sample[col] = max(0, base_sample[col])
                    
                    augmented_samples.append(base_sample)
                
                if augmented_samples:
                    augmented_df = pd.DataFrame(augmented_samples, columns=X.columns)
                    X_augmented = pd.concat([X_augmented, augmented_df], ignore_index=True)
                    y_augmented = np.concatenate([y_augmented, [class_idx] * len(augmented_samples)])
    
    return X_augmented, y_augmented


def apply_advanced_feature_engineering(X_balanced, original_columns):
    """Apply advanced feature engineering seperti di notebook"""
    
    X_balanced_df = pd.DataFrame(X_balanced, columns=original_columns)
    
    # 1. Log transformation untuk nilai moneter
    if 'Nilai_Token' in X_balanced_df.columns:
        X_balanced_df['Nilai_Token_log'] = np.log1p(X_balanced_df['Nilai_Token'])
        
        # Binning nilai proyek berdasarkan kuartil
        q1 = X_balanced_df['Nilai_Token'].quantile(0.25)
        q3 = X_balanced_df['Nilai_Token'].quantile(0.75)
        X_balanced_df['Nilai_Category_Small'] = (X_balanced_df['Nilai_Token'] < q1).astype(int)
        X_balanced_df['Nilai_Category_Large'] = (X_balanced_df['Nilai_Token'] > q3).astype(int)
    
    # 2. Interaction features
    if 'sektor_risk_score' in X_balanced_df.columns:
        if 'Nilai_Token' in X_balanced_df.columns:
            X_balanced_df['risk_value_interaction'] = X_balanced_df['sektor_risk_score'] * np.log1p(X_balanced_df['Nilai_Token'])
        if 'id_status' in X_balanced_df.columns:
            X_balanced_df['risk_status_interaction'] = X_balanced_df['sektor_risk_score'] * X_balanced_df['id_status']
    
    # 3. Squared terms untuk fitur numerik terpenting
    numeric_cols = X_balanced_df.select_dtypes(include=[np.number]).columns
    important_numeric = [col for col in numeric_cols if 'risk' in col.lower() or 'nilai' in col.lower()]
    
    if len(important_numeric) >= 1:
        for col in important_numeric[:3]:  # Top 3 most important
            X_balanced_df[f'{col}_squared'] = X_balanced_df[col] ** 2
    
    return X_balanced_df


def apply_feature_selection(X_balanced_df, y_balanced, top_k=15):
    """Apply correlation-based feature selection"""
    
    # Hitung korelasi setiap fitur dengan target
    feature_correlations = {}
    for col in X_balanced_df.columns:
        try:
            corr = np.corrcoef(X_balanced_df[col], y_balanced)[0, 1]
            if not np.isnan(corr):
                feature_correlations[col] = abs(corr)
        except:
            feature_correlations[col] = 0
    
    # Urutkan fitur berdasarkan korelasi absolut
    sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)
    
    # Pilih top k fitur
    n_features_to_select = min(top_k, len(sorted_features))
    selected_features = [feature for feature, _ in sorted_features[:n_features_to_select]]
    
    return X_balanced_df[selected_features]


def train_optimized_model(X_train_scaled, y_train):
    """Train model dengan hyperparameter tuning"""
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000, 2000],
        'class_weight': ['balanced']
    }
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"   Best parameters: {grid_search.best_params_}")
    print(f"   Best CV Score: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_


def evaluate_model_comprehensive(model, X_test_scaled, y_test, risk_classes):
    """Comprehensive model evaluation dengan bootstrap"""
    
    y_pred = model.predict(X_test_scaled)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Bootstrap confidence intervals
    bootstrap_scores = []
    for i in range(100):  # Reduced for faster training
        X_boot, y_boot = resample(X_test_scaled, y_test, random_state=i)
        y_pred_boot = model.predict(X_boot)
        score = f1_score(y_boot, y_pred_boot, average='weighted')
        bootstrap_scores.append(score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    ci_width = ci_upper - ci_lower
    ci_stability = 1 - (ci_width / f1_weighted) if f1_weighted > 0 else 0
    
    # Composite score
    composite_score = (0.4 * f1_weighted + 0.3 * accuracy + 0.2 * f1_macro + 0.1 * ci_stability)
    
    # Rating
    if composite_score >= 0.90:
        rating = "9.5/10 - Excellent"
    elif composite_score >= 0.85:
        rating = "9.0/10 - Very Good"
    elif composite_score >= 0.80:
        rating = "8.5/10 - Good"
    else:
        rating = f"{composite_score*10:.1f}/10 - Needs Improvement"
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'composite_score': composite_score,
        'rating': rating,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_stability': ci_stability
    }


def save_optimized_model(model, scaler, label_encoder, feature_columns, df_sektor, performance_metrics, model_dir):
    """Save all optimized model components"""
    
    model_files = {
        'optimized_model': f'{model_dir}/optimized_risk_prediction_model.joblib',
        'scaler': f'{model_dir}/optimized_risk_prediction_scaler.joblib',
        'label_encoder': f'{model_dir}/optimized_risk_prediction_label_encoder.joblib',
        'features': f'{model_dir}/optimized_risk_prediction_features.joblib',
        'metadata': f'{model_dir}/optimized_model_metadata.joblib'
    }
    
    # Save model components
    joblib.dump(model, model_files['optimized_model'])
    joblib.dump(scaler, model_files['scaler'])
    joblib.dump(label_encoder, model_files['label_encoder'])
    joblib.dump(feature_columns, model_files['features'])
    
    # Save metadata
    metadata = {
        'model_type': 'OptimizedLogisticRegression',
        'performance_metrics': performance_metrics,
        'feature_count': len(feature_columns),
        'hyperparameters': model.get_params() if hasattr(model, 'get_params') else 'N/A',
        'selected_features': feature_columns
    }
    joblib.dump(metadata, model_files['metadata'])
    
    if df_sektor is not None:
        model_files['sector_reference'] = f'{model_dir}/sector_reference.joblib'
        joblib.dump(df_sektor, model_files['sector_reference'])
    
    # Generate summary report
    summary_report = f"""
=== OPTIMIZED MODEL TRAINING SUMMARY ===
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

FINAL MODEL PERFORMANCE:
- Model Type: OptimizedLogisticRegression
- F1-weighted: {performance_metrics['f1_weighted']:.3f}
- Accuracy: {performance_metrics['accuracy']:.3f}
- Composite Score: {performance_metrics['composite_score']:.3f}
- Rating: {performance_metrics['rating']}

OPTIMIZATIONS APPLIED:
1. ✅ SMOTE-like data augmentation for class balance
2. ✅ Advanced feature engineering (log, interactions, squared)
3. ✅ Correlation-based feature selection
4. ✅ Hyperparameter tuning with GridSearchCV
5. ✅ Bootstrap confidence intervals

FEATURES SELECTED: {len(feature_columns)}
{chr(10).join([f"  {i+1}. {f}" for i, f in enumerate(feature_columns)])}

USAGE:
- Use optimized_* files for production
- Apply same preprocessing pipeline
- Check confidence scores for predictions
===================================================
"""
    
    report_path = f'{model_dir}/optimized_training_report.txt'
    with open(report_path, 'w') as f:
        f.write(summary_report)
    
    model_files['report'] = report_path
    return model_files

if __name__ == "__main__":
    train_and_save_risk_model()
