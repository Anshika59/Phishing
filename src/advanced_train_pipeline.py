import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost_model import get_xgboost_model
except ImportError:
    from src.xgboost_model import get_xgboost_model


def load_features():
    """
    Loads engineered feature matrix from data/processed/features_v2.csv.
    """
    features_path = os.path.join("data", "processed", "features_v2.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing {features_path}. Run Week 3 feature extraction first.")

    df = pd.read_csv(features_path)
    y = df["label"].astype(int)

    # Keep only numeric feature columns
    X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    return X, y


def run_advanced_comparison():
    print("\nLoading engineered features for Week 5 Model Comparison...")
    X, y = load_features()

    # Safe train/test split for small or large datasets
    min_class = y.value_counts().min()
    test_count = max(1, int(len(y) * 0.2))
    stratify_y = y if (min_class >= 2 and test_count >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": get_xgboost_model(),
    }

    # Configure safe cross-validation splits
    train_min_class = y_train.value_counts().min()
    n_splits = min(5, int(train_min_class)) if train_min_class >= 2 else 0

    results = []
    best_model = None
    best_model_name = ""
    best_score = -1.0

    for name, model in models.items():
        print(f"--> Training and evaluating {name}...")

        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_res = cross_validate(
                model, X_train, y_train, cv=cv, scoring=["accuracy", "f1"], error_score=0.0
            )
            cv_acc = float(np.mean(cv_res["test_accuracy"]))
            cv_f1 = float(np.mean(cv_res["test_f1"]))
        else:
            cv_acc, cv_f1 = 0.0, 0.0

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Model": name,
            "CV Accuracy": round(cv_acc, 4),
            "CV F1": round(cv_f1, 4),
            "Test Accuracy": round(test_acc, 4),
            "Test Precision": round(test_prec, 4),
            "Test Recall": round(test_rec, 4),
            "Test F1": round(test_f1, 4),
        })

        # Select optimal model (prioritize CV F1 / Test F1, preferring XGBoost on ties)
        selection_score = (cv_f1 + test_f1) / 2.0
        if selection_score >= best_score:
            best_score = selection_score
            best_model = model
            best_model_name = name

    summary_df = pd.DataFrame(results)

    print("\n" + "=" * 82)
    print("             WEEK 5 MODEL COMPARISON: XGBOOST VS. BASELINES")
    print("=" * 82)
    print(summary_df.to_string(index=False))
    print("=" * 82)

    os.makedirs("models", exist_ok=True)
    xgb_path = os.path.join("models", "xgboost_model.pkl")
    joblib.dump(models["XGBoost"], xgb_path)

    best_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, best_path)

    print(f"\nXGBoost artifact saved to {xgb_path}")
    print(f"Optimal Overall Model Selected: {best_model_name} -> Saved to {best_path}\n")


if __name__ == "__main__":
    run_advanced_comparison()