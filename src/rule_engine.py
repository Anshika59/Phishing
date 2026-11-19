import os
import joblib
import numpy as np
import pandas as pd

try:
    from reputation_checker import evaluate_url_reputation
    from auth_validator import evaluate_authentication
except ImportError:
    from src.reputation_checker import evaluate_url_reputation
    from src.auth_validator import evaluate_authentication


class HybridRuleEngine:
    """
    Integrates rule-based validation outcomes (blacklists, URL reputation,
    and SPF/DKIM/DMARC authentication) with ML phishing probability scores.
    """

    def __init__(self, ml_weight=0.65, rule_weight=0.35, enable_live_dnsbl=True):
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.enable_live_dnsbl = enable_live_dnsbl

    def compute_rule_score(self, email_text="", headers_or_raw=""):
        """
        Runs URL reputation and authentication checks and aggregates rule risk.
        """
        url_res = evaluate_url_reputation(email_text, enable_live_dnsbl=self.enable_live_dnsbl)
        auth_source = headers_or_raw if headers_or_raw else email_text
        auth_res = evaluate_authentication(auth_source)

        url_score = url_res["reputation_score"]
        auth_score = auth_res["auth_risk_score"]

        # Combined rule score prioritizes the strongest risk signal while accounting for both
        weighted_rule = (0.55 * url_score) + (0.45 * auth_score)
        combined_rule_score = round(min(1.0, max(weighted_rule, 0.85 * max(url_score, auth_score))), 4)

        all_flags = url_res["flags"] + auth_res["flags"]

        return {
            "combined_rule_score": combined_rule_score,
            "url_reputation": url_res,
            "authentication": auth_res,
            "triggered_rules": all_flags,
        }

    def fuse_scores(self, ml_probability, email_text="", headers_or_raw=""):
        """
        Adjusts an ML phishing probability [0.0, 1.0] using rule-based validation outcomes.
        """
        ml_prob = float(np.clip(ml_probability, 0.0, 1.0))
        rule_output = self.compute_rule_score(email_text=email_text, headers_or_raw=headers_or_raw)

        rule_score = rule_output["combined_rule_score"]
        is_blacklisted = rule_output["url_reputation"]["blacklisted"]
        hard_auth_fail = rule_output["authentication"]["hard_auth_fail"]
        auth_info = rule_output["authentication"]

        # Base weighted fusion
        fused_score = (self.ml_weight * ml_prob) + (self.rule_weight * rule_score)
        decision_reason = "Weighted ML + Rule fusion"

        # Rule Influence 1: Hard Blacklist Escalation (Spamhaus / Threat Feed hit)
        if is_blacklisted:
            fused_score = max(fused_score, 0.95)
            decision_reason = "Hard override: Domain/IP matched external blacklist"

        # Rule Influence 2: Hard Authentication Failure / Brand Spoofing Escalation
        elif hard_auth_fail and auth_info["auth_risk_score"] >= 0.70:
            fused_score = max(fused_score, 0.85)
            decision_reason = "Hard escalation: Critical SPF/DKIM/DMARC authentication failure"

        # Rule Influence 3: Moderate Rule Risk Boost
        elif rule_score >= 0.40:
            fused_score = min(1.0, max(fused_score, ml_prob + (0.30 * rule_score)))
            decision_reason = "Risk boost: Suspicious URL reputation or header mismatch"

        # Rule Influence 4: Verified Legitimate Sender Dampening (reduces false positives)
        elif (
            rule_score == 0.0
            and auth_info["spf"] == "pass"
            and auth_info["dkim"] == "pass"
            and auth_info["dmarc"] == "pass"
        ):
            fused_score = ml_prob * 0.85
            decision_reason = "Confidence dampening: SPF, DKIM, and DMARC all passed with clean URLs"

        final_score = round(float(np.clip(fused_score, 0.0, 1.0)), 4)

        if final_score >= 0.70:
            verdict = "PHISHING"
        elif final_score >= 0.40:
            verdict = "SUSPICIOUS"
        else:
            verdict = "LEGITIMATE"

        return {
            "ml_probability": round(ml_prob, 4),
            "rule_risk_score": rule_score,
            "final_hybrid_score": final_score,
            "verdict": verdict,
            "fusion_policy_applied": decision_reason,
            "triggered_rules": rule_output["triggered_rules"],
        }


def run_batch_validation_if_available(engine):
    """
    Runs the hybrid rule engine over processed dataset samples if available.
    """
    emails_path = os.path.join("data", "processed", "emails.csv")
    features_path = os.path.join("data", "processed", "features_v2.csv")
    model_path = os.path.join("models", "best_model.pkl")

    if not (os.path.exists(emails_path) and os.path.exists(features_path) and os.path.exists(model_path)):
        return

    df_emails = pd.read_csv(emails_path)
    df_feat = pd.read_csv(features_path)
    model = joblib.load(model_path)

    X = df_feat.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    if len(X) == 0 or len(X) != len(df_emails):
        return

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        ml_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    else:
        ml_probs = model.predict(X).astype(float)

    text_col = "text" if "text" in df_emails.columns else ("body" if "body" in df_emails.columns else None)
    if not text_col:
        return

    print("\n--- Dataset Hybrid Validation Summary ---")
    for idx in range(min(5, len(df_emails))):
        txt = str(df_emails.iloc[idx][text_col])
        res = engine.fuse_scores(ml_probability=ml_probs[idx], email_text=txt)
        print(
            f"Row {idx + 1}: ML={res['ml_probability']:.2f} | "
            f"Rule={res['rule_risk_score']:.2f} -> "
            f"Final={res['final_hybrid_score']:.2f} [{res['verdict']}]"
        )


if __name__ == "__main__":
    engine = HybridRuleEngine(enable_live_dnsbl=True)

    test_cases = [
        {
            "name": "Case 1: Verified Legitimate Notification (Borderline ML Dampened)",
            "ml_prob": 0.44,
            "headers": """From: GitHub <noreply@github.com>
Authentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass""",
            "body": "Your pull request was merged. View details at https://github.com/Anshika59/Phishing",
        },
        {
            "name": "Case 2: Polite Phishing Text with Blacklisted URL (Blacklist Override)",
            "ml_prob": 0.35,
            "headers": """From: Account Support <support@paypal-security-update.xyz>
Authentication-Results: mx.google.com; dkim=none; spf=softfail; dmarc=none""",
            "body": "Hello customer, kindly review your recent billing statement at https://paypal-security-update.xyz/signin",
        },
        {
            "name": "Case 3: Spoofed Sender Failing SPF/DKIM/DMARC (Auth Escalation)",
            "ml_prob": 0.48,
            "headers": """From: PayPal Security <alert@random-mailer.xyz>
Reply-To: verify@collect-credentials.top
Authentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail""",
            "body": "Please confirm your account details immediately via http://192.168.10.45/login",
        },
    ]

    print("\n" + "=" * 78)
    print("         WEEK 6: HYBRID RULE-BASED + ML SCORE FUSION ENGINE")
    print("=" * 78)

    for case in test_cases:
        outcome = engine.fuse_scores(
            ml_probability=case["ml_prob"],
            email_text=case["body"],
            headers_or_raw=case["headers"],
        )
        print(f"\n{case['name']}")
        print(f"  Raw ML Probability : {outcome['ml_probability']}")
        print(f"  Rule Risk Score    : {outcome['rule_risk_score']}")
        print(f"  Final Hybrid Score : {outcome['final_hybrid_score']}  -->  VERDICT: {outcome['verdict']}")
        print(f"  Fusion Policy      : {outcome['fusion_policy_applied']}")
        print(f"  Triggered Rules    : {outcome['triggered_rules']}")

    run_batch_validation_if_available(engine)
    print("\n" + "=" * 78 + "\n")