import os
import re
import json
import numpy as np
import pandas as pd

try:
    from hybrid_detector import HybridPhishingDetector
    from shap_explainer import TabularShapExplainer
    from lime_explainer import TextLimeExplainer
except ImportError:
    from src.hybrid_detector import HybridPhishingDetector
    from src.shap_explainer import TabularShapExplainer
    from src.lime_explainer import TextLimeExplainer


FEATURE_PLAIN_NAMES = {
    "url_count": "number of embedded links",
    "has_ip_in_url": "direct IP address used inside a link",
    "has_redirect_in_url": "URL redirection parameter",
    "has_risky_attachment": "high-risk attachment indicator",
    "urgency_count": "urgent or high-pressure phrasing",
    "exclamation_count": "frequent exclamation marks",
    "uppercase_ratio": "elevated ALL-CAPS text ratio",
    "char_len": "total email character length",
    "word_count": "total email word count",
}


class PlainLanguageExplainabilityEngine:
    """
    Aggregates SHAP, LIME, and rule-based validation attributions and translates
    them into plain-language explanations for end users.
    """

    def __init__(self):
        self.detector = HybridPhishingDetector(enable_live_dnsbl=False, refined_mode=True)
        self.shap_explainer = TabularShapExplainer()
        self.lime_explainer = TextLimeExplainer()

    def _extract_basic_features(self, email_text):
        """
        Builds a feature dictionary aligned with the tabular model's expected columns.
        """
        text = str(email_text or "")
        urls = re.findall(r"(?:https?://|www\.)[^\s<>\"'\]\)]+", text, flags=re.IGNORECASE)
        has_ip = int(any(re.search(r"https?://\d{1,3}(?:\.\d{1,3}){3}", u) for u in urls))
        has_redirect = int(any(k in u.lower() for u in urls for k in ("redirect=", "url=", "next=", "dest=")))
        alpha_chars = [c for c in text if c.isalpha()]
        upper_ratio = (sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)) if alpha_chars else 0.0

        row = {col: 0.0 for col in self.shap_explainer.feature_names}
        heuristics = {
            "url_count": float(len(urls)),
            "has_ip_in_url": float(has_ip),
            "has_redirect_in_url": float(has_redirect),
            "has_risky_attachment": 0.0,
            "exclamation_count": float(text.count("!")),
            "uppercase_ratio": round(upper_ratio, 4),
            "char_len": float(len(text)),
            "word_count": float(len(text.split())),
        }
        for k, v in heuristics.items():
            if k in row:
                row[k] = v
        return row

    def _translate_feature_name(self, raw_feature):
        if raw_feature in FEATURE_PLAIN_NAMES:
            return FEATURE_PLAIN_NAMES[raw_feature]
        if raw_feature.startswith("distilbert_emb_"):
            return "semantic language pattern embedding"
        return raw_feature.replace("_", " ")

    def _translate_rule_flag(self, flag):
        f_low = flag.lower()
        if "blacklist feed match" in f_low or "spamhaus" in f_low:
            return f"The linked domain appears on a known phishing blacklist ({flag.split(':')[-1].strip()})."
        if "direct ip address url" in f_low:
            return "The email links directly to a raw IP address instead of a legitimate domain name."
        if "high-risk tld" in f_low:
            return f"The link uses a top-level domain frequently abused in phishing campaigns ({flag})."
        if "suspicious url keywords" in f_low:
            return f"The web address contains credential-harvesting keywords ({flag})."
        if "spf authentication failed" in f_low:
            return "The sending server failed SPF verification, meaning it is not authorized to send for this domain."
        if "dkim signature verification failed" in f_low:
            return "The email's cryptographic DKIM signature failed verification, indicating possible tampering or spoofing."
        if "dmarc policy check failed" in f_low:
            return "The message failed DMARC domain alignment and anti-spoofing policy checks."
        if "reply-to domain mismatch" in f_low:
            return f"Replies are routed to a different external domain than the visible sender ({flag})."
        if "display name spoofing" in f_low:
            return f"The sender's display name impersonates a trusted brand while originating from an unrelated domain ({flag})."
        return f"Security rule triggered: {flag}."

    def explain_email(self, email_text, headers_or_raw="", feature_row=None):
        """
        Generates hybrid prediction, SHAP + LIME attributions, and a plain-language report.
        """
        if feature_row is None:
            feature_row = self._extract_basic_features(email_text)

        # 1. Hybrid prediction & rule outcomes
        hybrid_res = self.detector.predict(
            email_text=email_text,
            headers_or_raw=headers_or_raw,
            feature_row=feature_row,
        )

        # 2. SHAP tabular feature attributions
        shap_res = self.shap_explainer.explain_instance(feature_row, top_k=4)

        # 3. LIME local word attributions
        lime_res = self.lime_explainer.explain_text(email_text, num_features=5, num_samples=120)

        # 4. Map attributions to plain-language statements
        plain_reasons = []

        # Rule-based plain-language explanations
        for flag in hybrid_res["triggered_rules"]:
            plain_reasons.append(self._translate_rule_flag(flag))

        if not hybrid_res["triggered_rules"] and "passed" in hybrid_res["fusion_policy_applied"].lower():
            plain_reasons.append(
                "Sender authentication (SPF, DKIM, and DMARC) passed and all embedded links appear clean."
            )

        # LIME word-level plain-language explanations
        phish_tokens = [t["token"] for t in lime_res["token_attributions"] if t["lime_weight"] > 0]
        safe_tokens = [t["token"] for t in lime_res["token_attributions"] if t["lime_weight"] < 0]

        if phish_tokens:
            quoted = ", ".join(f"'{w}'" for w in phish_tokens[:3])
            plain_reasons.append(
                f"Text analysis (LIME) flagged words/phrases ({quoted}) that strongly resemble phishing lures."
            )
        if safe_tokens and hybrid_res["verdict"] == "LEGITIMATE":
            quoted = ", ".join(f"'{w}'" for w in safe_tokens[:3])
            plain_reasons.append(
                f"Text analysis (LIME) identified normal conversational terms ({quoted}) associated with legitimate mail."
            )

        # SHAP structural feature plain-language explanations
        top_shap = [a for a in shap_res["top_attributions"] if abs(a["shap_value"]) > 0]
        if top_shap:
            lead = top_shap[0]
            feat_desc = self._translate_feature_name(lead["feature"])
            direction = "increased" if lead["shap_value"] > 0 else "reduced"
            plain_reasons.append(
                f"Metadata attribution (SHAP) showed that the {feat_desc} (value={lead['feature_value']}) {direction} the risk score."
            )
        else:
            lead = shap_res["top_attributions"][0]
            feat_desc = self._translate_feature_name(lead["feature"])
            plain_reasons.append(
                f"Metadata attribution (SHAP) evaluated structural features such as {feat_desc} (value={lead['feature_value']})."
            )

        # Executive Plain-Language Summary
        verdict = hybrid_res["verdict"]
        score_pct = round(hybrid_res["final_hybrid_score"] * 100, 1)
        if verdict == "PHISHING":
            summary = (
                f"This email is classified as PHISHING ({score_pct}% risk score). "
                f"{plain_reasons[0]} Do not click any links or share credentials."
            )
        else:
            summary = (
                f"This email is classified as LEGITIMATE ({score_pct}% risk score). "
                f"{plain_reasons[0]}"
            )

        return {
            "verdict": verdict,
            "final_hybrid_score": hybrid_res["final_hybrid_score"],
            "ml_probability": hybrid_res["ml_probability"],
            "rule_risk_score": hybrid_res["rule_risk_score"],
            "plain_language_summary": summary,
            "plain_language_reasons": list(dict.fromkeys(plain_reasons)),
            "shap_top_features": shap_res["top_attributions"],
            "lime_top_tokens": lime_res["token_attributions"],
        }


if __name__ == "__main__":
    engine = PlainLanguageExplainabilityEngine()

    samples = [
        {
            "name": "Sample 1: Suspected Credential Phishing Email",
            "headers": (
                "From: Microsoft Security <alert@microsoft-auth-alert.top>\n"
                "Reply-To: harvest@collect-credentials.top\n"
                "Authentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail"
            ),
            "body": "URGENT: Your account has been suspended! Verify your password immediately at https://microsoft-auth-alert.top/login",
        },
        {
            "name": "Sample 2: Legitimate Repository Notification",
            "headers": (
                "From: GitHub <noreply@github.com>\n"
                "Authentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass"
            ),
            "body": "Hi team, pull request #45 has been merged. Review the sprint notes at https://github.com/Anshika59/Phishing",
        },
    ]

    saved_outputs = []

    print("\n" + "=" * 82)
    print("       WEEK 8: PLAIN-LANGUAGE EXPLAINABILITY REPORT (SHAP + LIME + RULES)")
    print("=" * 82)

    for s in samples:
        report = engine.explain_email(email_text=s["body"], headers_or_raw=s["headers"])
        saved_outputs.append({"sample": s["name"], **report})

        print(f"\n{s['name']}")
        print(f"  Verdict      : {report['verdict']} (Hybrid Score: {report['final_hybrid_score']})")
        print(f"  Summary      : {report['plain_language_summary']}")
        print("  Key Reasons  :")
        for idx, reason in enumerate(report["plain_language_reasons"], 1):
            print(f"    {idx}. {reason}")

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "week8_sample_explanations.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saved_outputs, f, indent=2)

    print(f"\nSaved plain-language explanation report to {out_path}")
    print("=" * 82 + "\n")