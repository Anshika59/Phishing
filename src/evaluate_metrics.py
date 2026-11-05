import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

def evaluate_cross_validation(model, X, y, cv_folds=5):
    """
    Performs cross-validation, automatically adapting strategy based on sample size.
    """
    class_counts = y.value_counts()
    min_class_samples = class_counts.min() if len(class_counts) > 0 else 0
    
    # Choose StratifiedKFold if sufficient samples exist per class, otherwise fallback to KFold
    if min_class_samples >= cv_folds and len(class_counts) > 1:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    else:
        n_splits = max(2, min(cv_folds, len(y)))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, error_score=0.0)
    
    return {
        "CV_Accuracy": cv_results['test_accuracy'].mean(),
        "CV_Precision": cv_results['test_precision'].mean(),
        "CV_Recall": cv_results['test_recall'].mean(),
        "CV_F1": cv_results['test_f1'].mean()
    }

def calculate_test_metrics(y_true, y_pred):
    """
    Computes accuracy, precision, recall, and F1-score on test set.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred)
    }