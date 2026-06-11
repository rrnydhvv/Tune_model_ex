import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def main():
    print("="*50)
    print("Toxicity Prediction - 50F (Machine Cleaned)")
    print("="*50)

    # 1. Load Data
    base_path = r"c:\Users\phucb\Documents\Code\Tune_model_ex\data\toxic"
    data_file = os.path.join(base_path, "Toxicity-Cleaned-50F.csv")

    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Preprocessing
    print(f"Original dataset shape: {df.shape}")

    # Label Encode 'Class' (Toxic=1, NonToxic=0)
    le = LabelEncoder()
    y = le.fit_transform(df['Class'].str.strip())
    
    toxic_class_idx = np.where(le.classes_ == 'Toxic')[0]
    if len(toxic_class_idx) > 0:
        toxic_label = toxic_class_idx[0]
        if toxic_label == 0:
            y = 1 - y
            print("Note: 'Toxic' mapped to 1, 'NonToxic' mapped to 0.")

    X = df.drop(columns=['Class'])

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Tuning Base Models with SMOTE in Pipeline
    print("\n--- Tuning Base Models with SMOTE ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_to_test = {
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                'classifier__kernel': ['linear', 'rbf'],
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': ['scale', 'auto', 0.1]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [3, 5, 7],
                'classifier__min_samples_split': [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
            "params": {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [2, 3, 5],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }

    best_base_models = {}

    for name, config in models_to_test.items():
        print(f"\nTuning {name} + SMOTE...")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(k_neighbors=5, random_state=42)),
            ('classifier', config['model'])
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=config['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        print(f" Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        print(f" Best Params: {grid_search.best_params_}")
        
        # Save the best classifier (we extract it from the pipeline for stacking)
        best_base_models[name] = grid_search.best_estimator_.named_steps['classifier']

    # 4. Assembling Stacking Classifier
    print("\n--- Assembling Stacking Classifier ---")
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('svm', best_base_models['SVM']),
            ('rf', best_base_models['Random Forest']),
            ('xgb', best_base_models['XGBoost'])
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5,
        n_jobs=-1
    )

    # Wrap the Stacking in SMOTE pipeline as well
    final_pipeline = ImbPipeline([
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('stack', stacking_clf)
    ])
    
    print("Training Stacking Classifier...")
    final_pipeline.fit(X_train_scaled, y_train)
    
    # 5. Threshold Tuning for Higher Recall
    print("\n--- Tuning Final Decision Threshold (Targeting Recall >= 0.85) ---")
    y_oof_proba = cross_val_predict(final_pipeline, X_train_scaled, y_train, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    
    best_t = 0.5
    best_prec = 0
    best_rec = 0
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for t in thresholds:
        y_oof_pred = (y_oof_proba >= t).astype(int)
        rec = recall_score(y_train, y_oof_pred)
        prec = precision_score(y_train, y_oof_pred, zero_division=0)
        
        if rec >= 0.85:
            if prec > best_prec:
                best_prec = prec
                best_rec = rec
                best_t = t
                
    if best_prec == 0:
        best_t = 0.5 # fallback
            
    print(f"Optimal Probability Threshold found: {best_t:.2f} (OOF Recall: {best_rec:.4f}, OOF Precision: {best_prec:.4f})")

    # 6. Final Evaluation on Holdout Test Set
    print("\n==================================================")
    print("EVALUATING STACKING MODEL ON UNSEEN TEST SET")
    print("==================================================")
    
    y_test_proba = final_pipeline.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = (y_test_proba >= 0.5).astype(int)
    y_pred_optimal = (y_test_proba >= best_t).astype(int)
    
    print(f"\n--- Default Threshold (0.50) ---")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_default):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_default):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_default):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred_default):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_default))
    
    print(f"\n--- Optimal Threshold ({best_t:.2f}) [HIGH RECALL TUNED] ---")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_optimal):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_optimal):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_optimal):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred_optimal):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_optimal))

if __name__ == "__main__":
    main()
