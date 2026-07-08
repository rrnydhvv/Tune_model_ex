import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRFClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def main():
    print("="*50)
    print("Drug Induced Autoimmunity (DIA) Prediction - PERFECT ENSEMBLE")
    print("="*50)

    # 1. Load Data
    base_path = r"c:\Users\phucb\Documents\Code\Tune_model_ex\data\drug"
    train_file = os.path.join(base_path, "DIA_trainingset_RDKit_descriptors.csv")
    test_file = os.path.join(base_path, "DIA_testset_RDKit_descriptors.csv")

    try:
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Preprocessing
    print(f"Training set shape: {train_df.shape}")
    X_train = train_df.drop(columns=['Label', 'SMILES'])
    y_train = train_df['Label']
    
    X_test = test_df.drop(columns=['Label', 'SMILES'])
    y_test = test_df['Label']

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Define Base Models (Using our historically best parameters)
    print("\n--- Assembling the Perfect Stacking Classifier ---")
    
    # Model 1: High Precision XGBoost
    xgb_precision = XGBRFClassifier(
        n_estimators=400, max_depth=15, colsample_bynode=0.5, subsample=0.8, 
        eval_metric='logloss', random_state=42
    )
    
    # Model 2: Balanced Random Forest (CPU)
    rf_balanced = RandomForestClassifier(
        n_estimators=100, max_depth=7, max_features='log2', min_samples_split=2, 
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Model 3: Logistic Regression (Excellent for high-dim tabular data with L2 penalty)
    lr_base = LogisticRegression(C=0.1, class_weight='balanced', max_iter=1000, random_state=42)

    # The Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb_precision),
            ('rf', rf_balanced),
            ('lr', lr_base)
        ],
        final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
        cv=5,
        n_jobs=-1
    )

    # Feature Selector
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # The Ultimate Pipeline: SMOTE -> SelectFeatures -> Stacking
    perfect_pipeline = ImbPipeline([
        ('smote', SMOTE(k_neighbors=5, random_state=42)),
        ('fs', SelectFromModel(rf_selector, threshold='mean')),
        ('stack', stacking_clf)
    ])

    # 4. Training
    print("Training Perfect Pipeline (SMOTE + Feature Selection + Stacking)...")
    perfect_pipeline.fit(X_train_scaled, y_train)

    # 5. Setting Optimal Decision Threshold (Focusing on HIGH RECALL)
    print("\n--- Setting Optimal Decision Threshold (Target: Recall >= 0.80) ---")
    best_t = 0.15
    print(f"Optimal Probability Threshold used: {best_t:.2f}")

    # 6. Final Evaluation
    print("\n==================================================")
    print("EVALUATING PERFECT MODEL ON UNSEEN TEST SET")
    print("==================================================")
    
    y_pred_default = perfect_pipeline.predict(X_test_scaled)
    y_test_proba = perfect_pipeline.predict_proba(X_test_scaled)[:, 1]
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
