import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def calculate_metrics(y_true, y_pred):
    """
    Computes key performance metrics for binary phishing classification.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "Confusion_Matrix": cm
    }

def print_evaluation_summary(results_dict):
    """
    Prints a formatted evaluation table comparing all trained models.
    """
    summary_data = []
    for model_name, metrics in results_dict.items():
        summary_data.append({
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1_Score"]
        })
    
    df_results = pd.DataFrame(summary_data)
    print("\n=======================================================")
    print("           BASELINE MODEL EVALUATION SUMMARY           ")
    print("=======================================================")
    print(df_results.to_string(index=False))
    print("=======================================================\n")
    return df_results