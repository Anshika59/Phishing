from xgboost import XGBClassifier

def get_xgboost_model():
    """
    Initializes and returns the XGBoost classifier for Week 5 evaluation.
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        eval_metric="logloss",
        random_state=42
    )
    return model