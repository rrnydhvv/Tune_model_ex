import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# For XGBoost if installed, otherwise fallback to sklearn's Gradient Boosting
try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False
    print("XGBoost is not installed. Using Sklearn's GradientBoostingClassifier as an alternative.")

def main():
    print("="*50)
    print("Drug Induced Autoimmunity (DIA) Prediction")
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
    print(f"Test set shape: {test_df.shape}")

    # The target is 'Label', and 'SMILES' is a string column to drop
    X_train = train_df.drop(columns=['Label', 'SMILES'])
    y_train = train_df['Label']
    
    X_test = test_df.drop(columns=['Label', 'SMILES'])
    y_test = test_df['Label']

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Feature Selection
    print("\nApplying Feature Selection (Random Forest feature importance)...")
    # We use a basic RF to pick features that have an importance > mean
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector = SelectFromModel(rf_selector, threshold='mean')
    selector.fit(X_train_scaled, y_train)

    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_features_count = X_train_selected.shape[1]
    print(f"Features reduced from {X_train.shape[1]} to {selected_features_count}")

    # 4. Modeling
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
    }

    if has_xgboost:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, eval_metric='logloss')
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)

    # Adding Stacking Classifier
    estimators = [
        ('rf', models["Random Forest"]),
        ('xgb', models["XGBoost"] if has_xgboost else models["Gradient Boosting"])
    ]
    models["Stacking Classifier"] = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    )

    # 5. Training & Evaluation
    print("\nTraining and Evaluating Models...")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_selected, y_train)
        
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC-AUC : {roc_auc:.4f}")
            
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
