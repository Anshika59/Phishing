import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from baseline_models import get_baseline_models
from evaluate_metrics import evaluate_cross_validation, calculate_test_metrics

def run_baseline_training():
    features_path = os.path.join("data", "processed", "features_v2.csv")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing {features_path}. Run Week 3 feature extraction first.")
    
    df = pd.read_csv(features_path)
    
    # Separate features from label
    X = df.drop(columns=["label"], errors="ignore")
    y = df["label"]
    
    # Check if safe to use stratification
    class_counts = y.value_counts()
    min_class_count = class_counts.min() if len(class_counts) > 0 else 0
    test_count = int(len(df) * 0.20)
    
    use_stratify = y if (min_class_count >= 2 and test_count >= len(class_counts)) else None
    
    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=use_stratify
    )
    
    models = get_baseline_models()
    summary = []
    
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    
    print("\nExecuting Week 4 Baseline Model Training & Cross-Validation...\n")
    
    for name, model in models.items():
        print(f"--> Training and evaluating {name}...")
        
        # Cross-validation
        cv_metrics = evaluate_cross_validation(model, X_train, y_train, cv_folds=5)
        
        # Train on split
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_metrics = calculate_test_metrics(y_test, y_pred)
        
        summary.append({
            "Model": name,
            "CV Accuracy": round(cv_metrics["CV_Accuracy"], 4),
            "CV F1": round(cv_metrics["CV_F1"], 4),
            "Test Accuracy": round(test_metrics["Accuracy"], 4),
            "Test Precision": round(test_metrics["Precision"], 4),
            "Test Recall": round(test_metrics["Recall"], 4),
            "Test F1": round(test_metrics["Test F1"] if "Test F1" in test_metrics else test_metrics["F1_Score"], 4)
        })
        
        if test_metrics["F1_Score"] >= best_f1:
            best_f1 = test_metrics["F1_Score"]
            best_model_name = name
            best_model_obj = model
            
    summary_df = pd.DataFrame(summary)
    print("\n=========================================================================")
    print("                 BASELINE EVALUATION SUMMARY                      ")
    print("=========================================================================")
    print(summary_df.to_string(index=False))
    print("=========================================================================\n")
    
    # Save best performing model
    save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model_obj, save_path)
    print(f"Optimal Model Selected: {best_model_name} (Test F1 Score: {best_f1:.4f})")
    print(f"Model successfully saved to {save_path}\n")

if __name__ == "__main__":
    run_baseline_training()