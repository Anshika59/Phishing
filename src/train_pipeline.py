import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from baseline_models import get_baseline_models
from evaluate_metrics import calculate_metrics, print_evaluation_summary

def run_training_pipeline():
    features_path = os.path.join("data", "processed", "features.csv")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature dataset not found at {features_path}. Run Week 2 pipeline first.")
    
    df = pd.read_csv(features_path)
    
    # Drop non-feature columns
    X = df.drop(columns=["label"], errors="ignore")
    y = df["label"]
    
    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = get_baseline_models()
    evaluation_results = {}
    trained_models = {}
    
    best_f1 = -1.0
    best_model_name = None
    best_model_obj = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, predictions)
        evaluation_results[name] = metrics
        trained_models[name] = model
        
        if metrics["F1_Score"] > best_f1:
            best_f1 = metrics["F1_Score"]
            best_model_name = name
            best_model_obj = model
            
    # Display comparison table
    print_evaluation_summary(evaluation_results)
    
    # Save best model
    save_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model_obj, save_path)
    print(f"Best Model selected: {best_model_name} (F1 Score: {best_f1})")
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    run_training_pipeline()