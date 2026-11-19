import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


class TabularShapExplainer:
    """
    Generates SHAP feature attributions for the tabular phishing classifier.
    """

    def __init__(
        self,
        model_path="models/best_model.pkl",
        features_path="data/processed/features_v2.csv",
    ):
        self.model = joblib.load(model_path)
        self.X_background, self.feature_names = self._load_background_data(features_path)
        self.explainer = self._init_explainer()

    def _load_background_data(self, features_path):
        if os.path.exists(features_path):
            df = pd.read_csv(features_path)
            X = df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        else:
            X = pd.DataFrame()

        if hasattr(self.model, "feature_names_in_"):
            cols = list(self.model.feature_names_in_)
            for c in cols:
                if c not in X.columns:
                    X[c] = 0.0
            X = X[cols]
        else:
            cols = list(X.columns)

        if len(X) == 0:
            X = pd.DataFrame(np.zeros((5, len(cols))), columns=cols)

        return X, cols

    def _predict_phishing_proba(self, data):
        df_in = pd.DataFrame(data, columns=self.feature_names)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df_in)
            return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return self.model.predict(df_in).astype(float)

    def _init_explainer(self):
        bg = self.X_background.iloc[: min(50, len(self.X_background))]
        try:
            return shap.TreeExplainer(self.model)
        except Exception:
            return shap.Explainer(self._predict_phishing_proba, bg)

    def explain_instance(self, feature_row, top_k=5):
        """
        Computes SHAP attributions for a single feature vector (Series, dict, or 1-row DataFrame).
        """
        if isinstance(feature_row, dict):
            df_row = pd.DataFrame([feature_row])
        elif isinstance(feature_row, pd.Series):
            df_row = feature_row.to_frame().T
        else:
            df_row = feature_row.copy()

        for c in self.feature_names:
            if c not in df_row.columns:
                df_row[c] = 0.0
        df_row = df_row[self.feature_names].astype(float)

        phish_prob = float(self._predict_phishing_proba(df_row)[0])

        try:
            raw_sv = self.explainer.shap_values(df_row)
            if isinstance(raw_sv, list):
                sv_arr = np.array(raw_sv[1] if len(raw_sv) > 1 else raw_sv[0])[0]
            elif np.array(raw_sv).ndim == 3:
                sv_arr = np.array(raw_sv)[0, :, 1]
            else:
                sv_arr = np.array(raw_sv)[0]
        except Exception:
            explanation = self.explainer(df_row)
            vals = explanation.values
            sv_arr = vals[0, :, 1] if vals.ndim == 3 else vals[0]

        attributions = []
        for idx, feat in enumerate(self.feature_names):
            val = float(df_row.iloc[0][feat])
            shap_val = float(sv_arr[idx])
            attributions.append({
                "feature": feat,
                "feature_value": round(val, 4),
                "shap_value": round(shap_val, 5),
                "impact": "increases_phishing_risk" if shap_val > 0 else "decreases_phishing_risk",
            })

        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "tabular_phishing_prob": round(phish_prob, 4),
            "top_attributions": attributions[:top_k],
            "all_attributions": attributions,
        }


if __name__ == "__main__":
    explainer = TabularShapExplainer()
    sample_count = min(2, len(explainer.X_background))

    print("\n" + "=" * 78)
    print("         WEEK 8: SHAP TABULAR FEATURE ATTRIBUTIONS")
    print("=" * 78)

    for i in range(sample_count):
        row = explainer.X_background.iloc[i]
        res = explainer.explain_instance(row, top_k=5)
        print(f"\nSample {i + 1} | Tabular Phishing Probability: {res['tabular_phishing_prob']}")
        for item in res["top_attributions"]:
            sign = "+" if item["shap_value"] >= 0 else ""
            print(
                f"  - {item['feature']:<24} (val={item['feature_value']:<8}) "
                f"-> SHAP: {sign}{item['shap_value']:.5f} [{item['impact']}]"
            )
    print("\n" + "=" * 78 + "\n")