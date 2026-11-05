# 🛡️ Hybrid AI Phishing Detection System

A multi-tiered Machine Learning and NLP system designed to detect phishing threats by combining email header metadata analysis, URL/attachment threat scanning, and DistilBERT contextual embeddings.

---

## ✨ Key Features

* **Header & Infrastructure Inspection:** Evaluates SPF/DKIM verification status, domain mismatches, and typosquatting.
* **Content & NLP Vectorization:** Analyzes body text using TF-IDF and Hugging Face DistilBERT (`distilbert-base-uncased`) contextual embeddings.
* **URL & Attachment Threat Analysis:** Identifies raw IP links, redirect chains, and suspicious attachment file extensions (`.exe`, `.js`, `.vbs`, etc.).
* **ML Classification Pipeline:** Trains, cross-validates, and serializes optimal classifiers (Logistic Regression & Random Forest).

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn
* **NLP & Deep Learning:** Hugging Face Transformers, PyTorch
* **Data Handling:** Pandas, NumPy
* **Version Control:** Git, GitHub

---

## 🚀 Quick Start

```bash
# 1. Clone repo and setup virtual environment
git clone [https://github.com/Anshika59/Phishing-.git](https://github.com/Anshika59/Phishing-.git)
cd Phishing-
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run feature extraction pipeline
python src/extract_advanced_features.py

# 3. Train models and save serialized artifact
python src/train_pipeline.py
