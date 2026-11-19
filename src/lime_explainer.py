import os
import warnings
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

warnings.filterwarnings("ignore")


class TextLimeExplainer:
    """
    Generates LIME local token-level explanations for the fine-tuned DistilBERT model.
    """

    def __init__(self, model_dir="models/distilbert_phishing"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["LEGITIMATE", "PHISHING"]
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.explainer = LimeTextExplainer(class_names=self.class_names)

    def predict_proba(self, texts, batch_size=32):
        """
        Batch probability wrapper required by LIME's perturbation sampler.
        Returns shape (N, 2) -> [P(LEGITIMATE), P(PHISHING)].
        """
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            # Ensure 2-column probability output [P(Legit), P(Phish)]
            if probs.shape[1] == 1:
                phish_p = probs[:, 0]
                probs = np.column_stack([1.0 - phish_p, phish_p])

            all_probs.append(probs)

        return np.vstack(all_probs)

    def explain_text(self, email_text, num_features=6, num_samples=120):
        """
        Generates local word-level attribution weights for an email text.
        Positive weights push toward PHISHING (class 1); negative toward LEGITIMATE (class 0).
        """
        base_probs = self.predict_proba([email_text])[0]
        phish_prob = float(base_probs[1])

        exp = self.explainer.explain_instance(
            text_instance=str(email_text),
            classifier_fn=self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )

        raw_weights = exp.as_list(label=1)
        token_attributions = []
        for token, weight in raw_weights:
            w = float(weight)
            token_attributions.append({
                "token": token,
                "lime_weight": round(w, 5),
                "impact": "increases_phishing_risk" if w > 0 else "decreases_phishing_risk",
            })

        return {
            "nlp_phishing_prob": round(phish_prob, 4),
            "token_attributions": token_attributions,
        }


if __name__ == "__main__":
    samples = [
        "URGENT: Your bank account has been suspended! Verify your password immediately at http://verify-login.xyz",
        "Hi team, please find the updated sprint notes and pull request link on GitHub.",
    ]

    lime_exp = TextLimeExplainer()

    print("\n" + "=" * 78)
    print("         WEEK 8: LIME LOCAL TEXT EXPLANATIONS (DISTILBERT)")
    print("=" * 78)

    for idx, text in enumerate(samples, 1):
        res = lime_exp.explain_text(text, num_features=6, num_samples=120)
        print(f"\nSample {idx} | DistilBERT Phishing Probability: {res['nlp_phishing_prob']}")
        print(f"Text: {text}")
        for item in res["token_attributions"]:
            sign = "+" if item["lime_weight"] >= 0 else ""
            print(
                f"  - Token '{item['token']:<16}' -> LIME Weight: {sign}{item['lime_weight']:.5f} "
                f"[{item['impact']}]"
            )
    print("\n" + "=" * 78 + "\n")