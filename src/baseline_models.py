import os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_baseline_models():
    """
    Initializes and returns a dictionary of baseline machine learning classifiers.
    """
    models = {
        "Naive_Bayes": GaussianNB(),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    return models