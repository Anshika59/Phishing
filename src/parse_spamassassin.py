import os
import pandas as pd

def parse_spamassassin(raw_dir):
    """
    Module written by Riya to process the SpamAssassin dataset.
    """
    emails = []
    labels = []
    
    print("[Riya's Module] Parsing SpamAssassin dataset...")
    
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if "spamassassin" in file_path.lower() or "spam" in file_path.lower():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        emails.append(f.read())
                        labels.append(1)
                except Exception:
                    pass

    return pd.DataFrame({"text": emails, "label": labels})