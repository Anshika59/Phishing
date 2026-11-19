import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

try:
    from rule_engine import HybridRuleEngine
except ImportError:
    from src.rule_engine import HybridRuleEngine


class HybridPhishingDetector:
    """
    Combines Tabular ML + DistilBERT NLP probabilities with Rule-Based
    Validation adjustments (blacklists, URL reputation, SPF/DKIM/DMARC).
    """

    def __init__(
        self,
        tabular_model_path="models/best_model.pkl",
        bert_model_dir="models/distilbert_phishing",
        ml_weight=0.65,
        rule_weight=0.35,
        threshold=0.55,
        enable_live_dnsbl=False,
        refined_mode=False,
    ):
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.threshold = threshold
        self.refined_mode = refined_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Tabular ML model (XGBoost / Random Forest / LogReg)
        self.tabular_model = None
        if os.path.exists(tabular_model_path):
            self.tabular_model = joblib.load(tabular_model_path)

        # Load Fine-Tuned DistilBERT model
        self.tokenizer = None
        self.bert_model = None
        if os.path.exists(bert_model_dir):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(bert_model_dir)
            self.bert_model = DistilBertForSequenceClassification.from_pretrained(bert_model_dir)
            self.bert_model.to(self.device)
            self.bert_model.eval()

        self.rule_engine = HybridRuleEngine(
            ml_weight=ml_weight,
            rule_weight=rule_weight,
            enable_live_dnsbl=enable_live_dnsbl,
        )

    def predict_ml_score(self, email_text="", feature_row=None):
        """
        Computes combined ML phishing probability from DistilBERT (NLP)
        and Tabular ML (metadata/structural features).
        """
        scores = []
        weights = []

        # 1. NLP Layer: Fine-tuned DistilBERT probability
        if self.bert_model is not None and self.tokenizer is not None and str(email_text).strip():
            inputs = self.tokenizer(
                str(email_text),
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.bert_model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                bert_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            scores.append(bert_prob)
            weights.append(0.60)

        # 2. Metadata Layer: Tabular ML probability
        if self.tabular_model is not None and feature_row is not None:
            if isinstance(feature_row, dict):
                df_row = pd.DataFrame([feature_row])
            elif isinstance(feature_row, pd.Series):
                df_row = feature_row.to_frame().T
            else:
                df_row = feature_row.copy()

            if hasattr(self.tabular_model, "feature_names_in_"):
                expected_cols = list(self.tabular_model.feature_names_in_)
                for col in expected_cols:
                    if col not in df_row.columns:
                        df_row[col] = 0.0
                df_row = df_row[expected_cols]
            else:
                df_row = df_row.select_dtypes(include=[np.number]).fillna(0)

            if hasattr(self.tabular_model, "predict_proba"):
                t_probs = self.tabular_model.predict_proba(df_row)
                tab_prob = float(t_probs[0, 1]) if t_probs.shape[1] > 1 else float(t_probs[0, 0])
            else:
                tab_prob = float(self.tabular_model.predict(df_row)[0])

            scores.append(tab_prob)
            weights.append(0.40)

        if not scores:
            return 0.50

        combined_ml = float(np.average(scores, weights=weights))
        return round(float(np.clip(combined_ml, 0.0, 1.0)), 4)

    def adjust_with_rules(self, ml_prob, email_text="", headers_or_raw=""):
        """
        Applies either Initial (v1) or Refined (v2) rule-based score adjustment.
        """
        if not self.refined_mode:
            base = self.rule_engine.fuse_scores(
                ml_probability=ml_prob,
                email_text=email_text,
                headers_or_raw=headers_or_raw,
            )
            final_score = base["final_hybrid_score"]
            verdict = "PHISHING" if final_score >= self.threshold else "LEGITIMATE"
            base["verdict"] = verdict
            base["predicted_label"] = 1 if final_score >= self.threshold else 0
            return base

        # Refined Score Adjustment Mechanism (v2):
        # Addresses False Negatives (corroborated weak signals) & False Positives (verified senders)
        rule_out = self.rule_engine.compute_rule_score(
            email_text=email_text, headers_or_raw=headers_or_raw
        )
        url_res = rule_out["url_reputation"]
        auth_res = rule_out["authentication"]
        rule_score = rule_out["combined_rule_score"]

        fused = (self.ml_weight * ml_prob) + (self.rule_weight * rule_score)
        policy = "Refined weighted ML + Rule fusion"

        # 1. Deterministic blacklist override
        if url_res["blacklisted"]:
            fused = max(fused, 0.96)
            policy = "Hard override: Blacklisted domain/IP"

        # 2. Critical authentication failure or brand spoofing
        elif auth_res["hard_auth_fail"] and auth_res["auth_risk_score"] >= 0.65:
            fused = max(fused, 0.88)
            policy = "Hard escalation: Critical auth failure / brand spoofing"

        # 3. Corroborated multi-signal escalation (catches subtle spear-phishing FNs)
        elif url_res["reputation_score"] >= 0.25 and auth_res["auth_risk_score"] >= 0.20:
            corroborated_boost = 0.28 * (url_res["reputation_score"] + auth_res["auth_risk_score"])
            fused = min(1.0, max(fused, ml_prob + corroborated_boost, 0.62))
            policy = "Corroborated boost: Suspicious URL + weak/unaligned sender auth"

        # 4. Single strong URL/Header anomaly boost
        elif rule_score >= 0.35:
            fused = min(1.0, max(fused, ml_prob + (0.32 * rule_score)))
            policy = "Single-signal boost: Elevated URL or header risk"

        # 5. Verified legitimate sender dampening (eliminates borderline FPs)
        elif (
            url_res["reputation_score"] == 0.0
            and auth_res["spf"] == "pass"
            and auth_res["dkim"] == "pass"
            and auth_res["dmarc"] == "pass"
        ):
            fused = ml_prob * 0.72
            policy = "Strong dampening: SPF/DKIM/DMARC passed + clean URLs"

        final_score = round(float(np.clip(fused, 0.0, 1.0)), 4)
        pred_label = 1 if final_score >= self.threshold else 0

        return {
            "ml_probability": round(ml_prob, 4),
            "rule_risk_score": rule_score,
            "final_hybrid_score": final_score,
            "verdict": "PHISHING" if pred_label == 1 else "LEGITIMATE",
            "predicted_label": pred_label,
            "fusion_policy_applied": policy,
            "triggered_rules": rule_out["triggered_rules"],
        }

    def predict(self, email_text="", headers_or_raw="", feature_row=None):
        ml_prob = self.predict_ml_score(email_text=email_text, feature_row=feature_row)
        return self.adjust_with_rules(
            ml_prob=ml_prob, email_text=email_text, headers_or_raw=headers_or_raw
        )


if __name__ == "__main__":
    detector = HybridPhishingDetector(enable_live_dnsbl=False, refined_mode=False)

    sample_headers = """From: Security Team <alert@microsoft-auth-alert.top>
Authentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail"""
    sample_body = "Action required: Verify your Microsoft 365 password at https://microsoft-auth-alert.top/login"

    result = detector.predict(email_text=sample_body, headers_or_raw=sample_headers)
    print("\n=== HYBRID DETECTOR INTEGRATION TEST ===")
    for k, v in result.items():
        print(f"  {k}: {v}")