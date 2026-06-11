import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("="*50)
    print("Toxicity Prediction (1203 Features Dataset)")
    print("="*50)

    # 1. Load Data
    base_path = r"c:\Users\phucb\Documents\Code\Tune_model_ex\data\toxic"
    data_file = os.path.join(base_path, "data.csv")

    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # 2. Preprocessing
    print(f"Dataset shape: {df.shape}")

    # Label Encode 'Class' (Toxic=1, NonToxic=0)
    # The class values are 'Toxic' and 'NonToxic' (or similar capitalization based on file inspection)
    le = LabelEncoder()
    y = le.fit_transform(df['Class'].str.strip())
    
    # We assume class 1 is Toxic and 0 is NonToxic. Let's check classes_
    toxic_class_idx = np.where(le.classes_ == 'Toxic')[0]
    if len(toxic_class_idx) > 0:
        toxic_label = toxic_class_idx[0]
        # if 'Toxic' is 0, we flip so Toxic is 1
        if toxic_label == 0:
            y = 1 - y
            print("Note: 'Toxic' mapped to 1, 'NonToxic' mapped to 0.")
    
    X = df.drop(columns=['Class'])

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Feature Selection with RFE
    # Because of curse of dimensionality (1203 features vs 171 samples), we select top 20 features
    n_features_to_select = 20
    print(f"\nApplying RFE to reduce features from {X.shape[1]} to {n_features_to_select}...")
    
    # Using Logistic Regression with L1 penalty (Lasso equivalent) as the estimator for RFE
    # liblinear solver is good for small datasets and L1 penalty
    estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=50) # step > 1 speeds up RFE
    
    X_selected = rfe.fit_transform(X_scaled, y)
    
    # Let's print the selected features if column names exist
    selected_indices = np.where(rfe.support_)[0]
    selected_columns = X.columns[selected_indices]
    print(f"Selected Features: {list(selected_columns)}")

    # 4. Modeling & 5. Evaluation with Cross-validation
    models = {
        "Logistic Regression (Lasso/L1)": LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        "Logistic Regression (Ridge/L2)": LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
        "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
    }

    print("\nTraining and Evaluating Models with 5-Fold Stratified CV...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'roc_auc', 'f1_macro']
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        scores = cross_validate(model, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        
        acc_mean = scores['test_accuracy'].mean()
        roc_auc_mean = scores['test_roc_auc'].mean()
        f1_mean = scores['test_f1_macro'].mean()
        
        print(f"Mean Accuracy: {acc_mean:.4f}")
        print(f"Mean ROC-AUC : {roc_auc_mean:.4f}")
        print(f"Mean F1-Macro: {f1_mean:.4f}")

if __name__ == "__main__":
    main()
