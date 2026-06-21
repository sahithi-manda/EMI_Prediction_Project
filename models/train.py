import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

def train_and_evaluate():
    # 1. Load the dataset
    data_path = os.path.join('data', 'emi_dataset.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")
        
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    
    # Define features and targets
    X = df.drop(columns=['EMI_Eligibility', 'Max_Affordable_EMI'])
    y_class = df['EMI_Eligibility']
    y_reg = df['Max_Affordable_EMI']
    
    # Identify numerical and categorical columns
    num_cols = ['Age', 'Monthly_Income', 'Monthly_Expenses', 'Credit_Score', 
                'Existing_Loan_Amount', 'Existing_EMI', 'Dependents']
    cat_cols = ['Employment_Type']
    
    # Define Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
        ]
    )
    
    # 2. Train-Test Split (80% Train, 20% Test)
    # Using different random states or the same to keep subsets aligned
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42, stratify=y_class)
    
    # -------------------------------------------------------------------------
    # PART 1: CLASSIFICATION (EMI Eligibility)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING CLASSIFICATION MODELS (EMI Eligibility)")
    print("="*50)
    
    # Model A: Logistic Regression Pipeline
    clf_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    # Model B: Random Forest Classifier Pipeline
    clf_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10))
    ])
    
    # Train Logistic Regression
    clf_lr.fit(X_train, y_train_class)
    y_pred_lr = clf_lr.predict(X_test)
    
    # Train Random Forest
    clf_rf.fit(X_train, y_train_class)
    y_pred_rf = clf_rf.predict(X_test)
    
    # Metrics
    metrics_lr = {
        'Accuracy': accuracy_score(y_test_class, y_pred_lr),
        'Precision': precision_score(y_test_class, y_pred_lr),
        'Recall': recall_score(y_test_class, y_pred_lr),
        'F1-Score': f1_score(y_test_class, y_pred_lr)
    }
    
    metrics_rf = {
        'Accuracy': accuracy_score(y_test_class, y_pred_rf),
        'Precision': precision_score(y_test_class, y_pred_rf),
        'Recall': recall_score(y_test_class, y_pred_rf),
        'F1-Score': f1_score(y_test_class, y_pred_rf)
    }
    
    print("\n--- Logistic Regression Metrics ---")
    for k, v in metrics_lr.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test_class, y_pred_lr))
    
    print("\n--- Random Forest Classifier Metrics ---")
    for k, v in metrics_rf.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix (Random Forest):")
    print(confusion_matrix(y_test_class, y_pred_rf))
    
    # Select best classifier
    best_clf = clf_rf if metrics_rf['F1-Score'] > metrics_lr['F1-Score'] else clf_lr
    best_clf_name = "Random Forest Classifier" if best_clf == clf_rf else "Logistic Regression"
    print(f"\nSelected Best Classifier: {best_clf_name}")
    
    # Save best classifier
    os.makedirs('models', exist_ok=True)
    clf_path = os.path.join('models', 'best_classifier.joblib')
    joblib.dump(best_clf, clf_path)
    print(f"Saved {best_clf_name} to {clf_path}")
    
    # Save classifier performance stats for Streamlit
    clf_stats = {
        'lr_metrics': metrics_lr,
        'rf_metrics': metrics_rf,
        'best_name': best_clf_name,
        'confusion_matrix_lr': confusion_matrix(y_test_class, y_pred_lr).tolist(),
        'confusion_matrix_rf': confusion_matrix(y_test_class, y_pred_rf).tolist()
    }
    joblib.dump(clf_stats, os.path.join('models', 'classifier_stats.joblib'))
    
    # -------------------------------------------------------------------------
    # PART 2: REGRESSION (Maximum Affordable EMI)
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("TRAINING REGRESSION MODELS (Maximum Affordable EMI)")
    print("="*50)
    
    # Model A: Linear Regression Pipeline
    reg_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Model B: Random Forest Regressor Pipeline
    reg_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12))
    ])
    
    # Train Linear Regression
    reg_lr.fit(X_train, y_train_reg)
    y_pred_reg_lr = reg_lr.predict(X_test)
    
    # Train Random Forest Regressor
    reg_rf.fit(X_train, y_train_reg)
    y_pred_reg_rf = reg_rf.predict(X_test)
    
    # Metrics
    def calc_reg_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
    metrics_reg_lr = calc_reg_metrics(y_test_reg, y_pred_reg_lr)
    metrics_reg_rf = calc_reg_metrics(y_test_reg, y_pred_reg_rf)
    
    print("\n--- Linear Regression Metrics ---")
    for k, v in metrics_reg_lr.items():
        print(f"{k}: {v:.4f}")
        
    print("\n--- Random Forest Regressor Metrics ---")
    for k, v in metrics_reg_rf.items():
        print(f"{k}: {v:.4f}")
        
    # Select best regressor based on R2 score
    best_reg = reg_rf if metrics_reg_rf['R2'] > metrics_reg_lr['R2'] else reg_lr
    best_reg_name = "Random Forest Regressor" if best_reg == reg_rf else "Linear Regression"
    print(f"\nSelected Best Regressor: {best_reg_name}")
    
    # Save best regressor
    reg_path = os.path.join('models', 'best_regressor.joblib')
    joblib.dump(best_reg, reg_path)
    print(f"Saved {best_reg_name} to {reg_path}")
    
    # Save regressor performance stats for Streamlit
    reg_stats = {
        'lr_metrics': metrics_reg_lr,
        'rf_metrics': metrics_reg_rf,
        'best_name': best_reg_name
    }
    joblib.dump(reg_stats, os.path.join('models', 'regressor_stats.joblib'))
    
    # -------------------------------------------------------------------------
    # PART 3: FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    # Extract feature importance from Random Forest Classifier and Regressor
    # Get features after One-Hot Encoding
    ohe_categories = clf_rf.named_steps['preprocessor'].named_transformers_['cat'].categories_
    encoded_cat_features = []
    for col, categories in zip(cat_cols, ohe_categories):
        # We dropped first category
        encoded_cat_features.extend([f"{col}_{cat}" for cat in categories[1:]])
        
    feature_names = num_cols + encoded_cat_features
    
    rf_clf_importance = clf_rf.named_steps['classifier'].feature_importances_
    rf_reg_importance = reg_rf.named_steps['classifier' if 'classifier' in reg_rf.named_steps else 'regressor'].feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Classifier_Importance': rf_clf_importance,
        'Regressor_Importance': rf_reg_importance
    }).sort_values(by='Classifier_Importance', ascending=False)
    
    joblib.dump(importance_df, os.path.join('models', 'feature_importance.joblib'))
    print("\nFeature Importances:")
    print(importance_df)

if __name__ == '__main__':
    train_and_evaluate()
