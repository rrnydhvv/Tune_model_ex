import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier

def main():
    print("="*50)
    print("DATA CLEANING PIPELINE: 1204F -> 50F")
    print("="*50)

    # 1. Load Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "data", "toxic")
    data_file = os.path.join(base_path, "data.csv")

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"Original shape: {df.shape}")

    X = df.drop(columns=['Class'])
    y_raw = df['Class']
    y = y_raw.apply(lambda x: 1 if str(x).strip() == 'Toxic' else 0)

    # Step 1: Variance Threshold (remove constant features)
    print("\nStep 1: Removing constant features...")
    var_thres = VarianceThreshold(threshold=0.0)
    var_thres.fit(X)
    X_var = X.loc[:, var_thres.get_support()]
    print(f"Features remaining after Variance Filter: {X_var.shape[1]}")

    # Step 2: Correlation Filter (> 0.9)
    print("\nStep 2: Removing highly correlated features (>0.9)...")
    corr_matrix = X_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    X_corr = X_var.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} correlated features.")
    print(f"Features remaining after Correlation Filter: {X_corr.shape[1]}")

    # Step 3: Machine Learning Feature Importance (Top 50)
    print("\nStep 3: Selecting Top 50 features using Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_corr, y)
    
    importances = pd.Series(rf.feature_importances_, index=X_corr.columns)
    top_50_features = importances.sort_values(ascending=False).head(50).index.tolist()
    
    X_final = X_corr[top_50_features]
    print(f"Final shape of X: {X_final.shape}")

    # 4. Save to CSV
    df_final = pd.concat([X_final, y_raw], axis=1)
    output_file = os.path.join(base_path, "Toxicity-Cleaned-50F.csv")
    df_final.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved cleaned dataset to: {output_file}")

if __name__ == "__main__":
    main()
