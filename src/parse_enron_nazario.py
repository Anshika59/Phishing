import os
import pandas as pd

def parse_enron_and_nazario(raw_dir):
    """
    Module written by Arushi to process Enron (Legitimate) and Nazario (Phishing) data.
    """
    emails = []
    labels = []
    
    print("[Arushi's Module] Parsing Nazario & Enron datasets...")
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "nazario" in file_path.lower() or "phish" in file_path.lower():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        emails.append(f.read())
                        labels.append(1)
                except Exception:
                    pass
            elif "enron" in file_path.lower() or "legit" in file_path.lower():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        emails.append(f.read())
                        labels.append(0)
                except Exception:
                    pass
                    
    return pd.DataFrame({"text": emails, "label": labels})