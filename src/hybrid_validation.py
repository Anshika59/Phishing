import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from hybrid_detector import HybridPhishingDetector
except ImportError:
    from src.hybrid_detector import HybridPhishingDetector


VALIDATION_SAMPLES = [
    # --- Legitimate Emails (label = 0) ---
    {
        "id": "VAL-01",
        "label": 0,
        "category": "Legitimate Dev Notification",
        "headers": "From: GitHub <noreply@github.com>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "Pull request #42 was merged into main. Review changes at https://github.com/Anshika59/Phishing",
    },
    {
        "id": "VAL-02",
        "label": 0,
        "category": "Legitimate Team Meeting",
        "headers": "From: Anshika <2023a7r059@mietjammu.in>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "Hi Arushi and Riya, let's meet in the lab at 2 PM to review our Week 7 hybrid validation metrics.",
    },
    {
        "id": "VAL-03",
        "label": 0,
        "category": "Legitimate Urgent Billing Alert (FP Edge Case)",
        "headers": "From: AWS Billing <no-reply@amazonaws.com>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "URGENT ACTION REQUIRED: Your monthly cloud invoice is overdue and payment failed. Immediate attention required to avoid service suspension. View invoice at https://aws.amazon.com/billing",
    },
    {
        "id": "VAL-04",
        "label": 0,
        "category": "Legitimate Newsletter",
        "headers": "From: Python Weekly <newsletter@python.org>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "Read this week's top tutorials on scikit-learn pipelines and transformer optimization at https://www.python.org/blogs/",
    },
    {
        "id": "VAL-05",
        "label": 0,
        "category": "Legitimate Security Receipt",
        "headers": "From: Google <no-reply@accounts.google.com>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "Your password was changed recently. If this was you, no further action is needed.",
    },
    {
        "id": "VAL-06",
        "label": 0,
        "category": "Legitimate Urgent Deadline Reminder (FP Edge Case)",
        "headers": "From: Exam Cell <notifications@mietjammu.in>\nAuthentication-Results: mx.google.com; dkim=pass; spf=pass; dmarc=pass",
        "body": "URGENT FINAL WARNING: Submit your verified project report immediately before midnight or your account submission portal will close.",
    },

    # --- Phishing Emails (label = 1) ---
    {
        "id": "VAL-07",
        "label": 1,
        "category": "Blacklisted Domain Phishing",
        "headers": "From: PayPal Support <alert@paypal-security-update.xyz>\nAuthentication-Results: mx.google.com; dkim=none; spf=softfail; dmarc=none",
        "body": "Your PayPal account has been limited. Confirm your credentials at https://paypal-security-update.xyz/signin",
    },
    {
        "id": "VAL-08",
        "label": 1,
        "category": "Direct IP Credential Harvester",
        "headers": "From: IT Helpdesk <admin@corp-support.top>\nAuthentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail",
        "body": "Mandatory password reset required today: http://192.168.10.45/login/verify-account",
    },
    {
        "id": "VAL-09",
        "label": 1,
        "category": "Brand Spoofing + Auth Failure",
        "headers": "From: Microsoft Security <security@random-mailer.xyz>\nReply-To: harvest@collect-credentials.top\nAuthentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail",
        "body": "Unusual sign-in activity detected on your Microsoft account. Verify immediately.",
    },
    {
        "id": "VAL-10",
        "label": 1,
        "category": "Subtle Spear-Phishing Lure (FN Edge Case)",
        "headers": "From: Vendor Ops <ops@partner-portal.com>\nReply-To: ops@external-relay.net\nAuthentication-Results: mx.google.com; spf=softfail; dkim=none; dmarc=none",
        "body": "Hi team, sharing the updated Q4 project schedule document for your review here: https://drive-share-docs.icu/review",
    },
    {
        "id": "VAL-11",
        "label": 1,
        "category": "Subtle Invoice Redirect (FN Edge Case)",
        "headers": "From: Finance Desk <billing@supplier-gateway.com>\nReturn-Path: <bounce@unverified-sender.org>\nAuthentication-Results: mx.google.com; spf=softfail; dkim=none; dmarc=none",
        "body": "Hello, please find the attached remittance reference and shared folder link at https://cloud-doc-viewer.work/Q4-statement",
    },
    {
        "id": "VAL-12",
        "label": 1,
        "category": "Bank Verification Phishing",
        "headers": "From: Bank Alert <service@bankofamerica-verify-user.ml>\nAuthentication-Results: mx.google.com; dkim=fail; spf=fail; dmarc=fail",
        "body": "Security alert: Your online banking access is suspended. Restore access at https://bankofamerica-verify-user.ml/login",
    },
]


def evaluate_validation_set(refined_mode=False):
    detector = HybridPhishingDetector(enable_live_dnsbl=False, refined_mode=refined_mode)
    records = []

    for item in VALIDATION_SAMPLES:
        res = detector.predict(
            email_text=item["body"],
            headers_or_raw=item["headers"],
        )
        y_true = item["label"]
        y_pred = res["predicted_label"]

        if y_true == 1 and y_pred == 1:
            outcome_type = "TP"
        elif y_true == 0 and y_pred == 0:
            outcome_type = "TN"
        elif y_true == 0 and y_pred == 1:
            outcome_type = "FP"
        else:
            outcome_type = "FN"

        records.append({
            "id": item["id"],
            "category": item["category"],
            "y_true": y_true,
            "y_pred": y_pred,
            "outcome": outcome_type,
            "ml_prob": res["ml_probability"],
            "rule_score": res["rule_risk_score"],
            "final_score": res["final_hybrid_score"],
            "policy": res["fusion_policy_applied"],
            "flags": res["triggered_rules"],
        })

    df = pd.DataFrame(records)
    metrics = {
        "accuracy": round(float(accuracy_score(df["y_true"], df["y_pred"])), 4),
        "precision": round(float(precision_score(df["y_true"], df["y_pred"], zero_division=0)), 4),
        "recall": round(float(recall_score(df["y_true"], df["y_pred"], zero_division=0)), 4),
        "f1_score": round(float(f1_score(df["y_true"], df["y_pred"], zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(df["y_true"], df["y_pred"]).tolist(),
    }
    return df, metrics


if __name__ == "__main__":
    df_results, metrics = evaluate_validation_set(refined_mode=False)

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "week7_initial_validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "predictions": df_results.to_dict(orient="records")},
            f,
            indent=2,
        )

    print("\n" + "=" * 82)
    print("        WEEK 7: INITIAL HYBRID MODEL VALIDATION & ERROR ANALYSIS (v1)")
    print("=" * 82)
    print(f"Validation Samples : {len(df_results)} (6 Legitimate, 6 Phishing)")
    print(
        f"Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | F1-Score: {metrics['f1_score']:.4f}"
    )

    print("\n--- Per-Sample Validation Predictions ---")
    summary_cols = ["id", "y_true", "y_pred", "outcome", "ml_prob", "rule_score", "final_score", "category"]
    print(df_results[summary_cols].to_string(index=False))

    fps = df_results[df_results["outcome"] == "FP"]
    fns = df_results[df_results["outcome"] == "FN"]

    print("\n--- Error Analysis: False Positives (Legitimate Flagged as Phishing) ---")
    if fps.empty:
        print("None")
    else:
        for _, row in fps.iterrows():
            print(f"[{row['id']}] {row['category']} -> ML={row['ml_prob']}, Rule={row['rule_score']}, Final={row['final_score']}")
            print(f"   Root Cause: High ML urgency score insufficiently dampened despite verified SPF/DKIM/DMARC pass.")

    print("\n--- Error Analysis: False Negatives (Phishing Missed as Legitimate) ---")
    if fns.empty:
        print("None")
    else:
        for _, row in fns.iterrows():
            print(f"[{row['id']}] {row['category']} -> ML={row['ml_prob']}, Rule={row['rule_score']}, Final={row['final_score']}")
            print(f"   Root Cause: Polite spear-phishing text lowered ML score; weak auth + suspicious TLD needed corroborated boost.")

    print(f"\nSaved initial validation report to {report_path}")
    print("=" * 82 + "\n")