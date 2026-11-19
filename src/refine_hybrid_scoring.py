import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from hybrid_detector import HybridPhishingDetector
    from hybrid_validation import VALIDATION_SAMPLES
except ImportError:
    from src.hybrid_detector import HybridPhishingDetector
    from src.hybrid_validation import VALIDATION_SAMPLES


# Additional stress-test boundary cases to verify FP/FN resilience
STRESS_TEST_CASES = [
    {
        "id": "STRESS-FP",
        "label": 0,
        "category": "High-Urgency Verified Bank Statement (FP Stress)",
        "simulated_ml": 0.76,
        "headers": "From: Chase Alerts <no-reply@chase.com>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "URGENT: Unusual login attempt blocked. Verify your account statement immediately.",
    },
    {
        "id": "STRESS-FN",
        "label": 1,
        "category": "Polite Spear-Phishing + Weak Auth (FN Stress)",
        "simulated_ml": 0.36,
        "headers": "From: HR Benefits <hr@company-benefits-portal.com>\nReply-To: hr@external-mail-relay.net\nAuthentication-Results: mx.google.com; spf=softfail; dkim=none; dmarc=none",
        "body": "Hi, please review the updated employee handbook at https://hr-docs-portal.icu/view",
    },
]


def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def run_refinement_benchmark():
    threshold = 0.55
    detector_v1 = HybridPhishingDetector(enable_live_dnsbl=False, refined_mode=False, threshold=threshold)
    detector_v2 = HybridPhishingDetector(enable_live_dnsbl=False, refined_mode=True, threshold=threshold)

    rows = []

    # 1. Evaluate on the 12 Week 7 Validation Samples
    for item in VALIDATION_SAMPLES:
        ml_prob = detector_v1.predict_ml_score(email_text=item["body"])
        res_v1 = detector_v1.adjust_with_rules(ml_prob, email_text=item["body"], headers_or_raw=item["headers"])
        res_v2 = detector_v2.adjust_with_rules(ml_prob, email_text=item["body"], headers_or_raw=item["headers"])

        rows.append({
            "id": item["id"],
            "y_true": item["label"],
            "ml_score": ml_prob,
            "ml_pred": 1 if ml_prob >= 0.50 else 0,
            "v1_score": res_v1["final_hybrid_score"],
            "v1_pred": res_v1["predicted_label"],
            "v2_score": res_v2["final_hybrid_score"],
            "v2_pred": res_v2["predicted_label"],
            "v2_policy": res_v2["fusion_policy_applied"],
        })

    # 2. Evaluate on Stress-Test Edge Cases
    for item in STRESS_TEST_CASES:
        ml_prob = item["simulated_ml"]
        res_v1 = detector_v1.adjust_with_rules(ml_prob, email_text=item["body"], headers_or_raw=item["headers"])
        res_v2 = detector_v2.adjust_with_rules(ml_prob, email_text=item["body"], headers_or_raw=item["headers"])

        rows.append({
            "id": item["id"],
            "y_true": item["label"],
            "ml_score": ml_prob,
            "ml_pred": 1 if ml_prob >= 0.50 else 0,
            "v1_score": res_v1["final_hybrid_score"],
            "v1_pred": res_v1["predicted_label"],
            "v2_score": res_v2["final_hybrid_score"],
            "v2_pred": res_v2["predicted_label"],
            "v2_policy": res_v2["fusion_policy_applied"],
        })

    df = pd.DataFrame(rows)

    # Metrics comparison across all 14 evaluation samples
    m_ml = compute_metrics(df["y_true"], df["ml_pred"])
    m_v1 = compute_metrics(df["y_true"], df["v1_pred"])
    m_v2 = compute_metrics(df["y_true"], df["v2_pred"])

    summary_table = pd.DataFrame([
        {"Model Stage": "1. ML-Only Baseline", **m_ml,
         "FP": int(((df["y_true"] == 0) & (df["ml_pred"] == 1)).sum()),
         "FN": int(((df["y_true"] == 1) & (df["ml_pred"] == 0)).sum())},
        {"Model Stage": "2. Initial Hybrid (v1)", **m_v1,
         "FP": int(((df["y_true"] == 0) & (df["v1_pred"] == 1)).sum()),
         "FN": int(((df["y_true"] == 1) & (df["v1_pred"] == 0)).sum())},
        {"Model Stage": "3. Refined Hybrid (v2)", **m_v2,
         "FP": int(((df["y_true"] == 0) & (df["v2_pred"] == 1)).sum()),
         "FN": int(((df["y_true"] == 1) & (df["v2_pred"] == 0)).sum())},
    ])

    # Decision boundary separation metrics
    legit_v1 = df[df["y_true"] == 0]["v1_score"]
    phish_v1 = df[df["y_true"] == 1]["v1_score"]
    legit_v2 = df[df["y_true"] == 0]["v2_score"]
    phish_v2 = df[df["y_true"] == 1]["v2_score"]

    margin_v1 = round(float(phish_v1.min() - legit_v1.max()), 4)
    margin_v2 = round(float(phish_v2.min() - legit_v2.max()), 4)

    # Save refined hybrid configuration for Week 8+ deployment
    os.makedirs("models", exist_ok=True)
    config_path = os.path.join("models", "hybrid_config.json")
    hybrid_config = {
        "version": "2.0-refined",
        "ml_weight": 0.65,
        "rule_weight": 0.35,
        "decision_threshold": threshold,
        "blacklist_floor": 0.96,
        "critical_auth_floor": 0.88,
        "corroborated_signal_floor": 0.62,
        "verified_sender_dampening_factor": 0.72,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hybrid_config, f, indent=2)

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "week7_refined_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_table.to_dict(orient="records"),
            "class_separation_margin_v1": margin_v1,
            "class_separation_margin_v2": margin_v2,
            "details": rows,
        }, f, indent=2)

    print("\n" + "=" * 86)
    print("       WEEK 7: SCORE REFINEMENT & HYBRID PERFORMANCE BENCHMARK (v1 vs v2)")
    print("=" * 86)
    print(summary_table.to_string(index=False))

    print("\n--- Class Separation & Safety Margin Analysis ---")
    print(f"Initial Hybrid (v1) Legit Max={legit_v1.max():.4f} | Phish Min={phish_v1.min():.4f} | Separation Gap={margin_v1:+.4f}")
    print(f"Refined Hybrid (v2) Legit Max={legit_v2.max():.4f} | Phish Min={phish_v2.min():.4f} | Separation Gap={margin_v2:+.4f}")

    print("\n--- Stress-Test Boundary Corrections (v1 -> v2) ---")
    stress_df = df[df["id"].str.startswith("STRESS")]
    for _, r in stress_df.iterrows():
        print(
            f"[{r['id']}] True={r['y_true']} | ML={r['ml_score']:.2f} | "
            f"v1={r['v1_score']:.4f} (Pred={r['v1_pred']}) -> "
            f"v2={r['v2_score']:.4f} (Pred={r['v2_pred']}) | {r['v2_policy']}"
        )

    print(f"\nRefined hybrid config saved to {config_path}")
    print(f"Benchmark report saved to {report_path}")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    run_refinement_benchmark()