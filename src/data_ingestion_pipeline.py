import os
import pandas as pd
from parse_enron_nazario import parse_enron_and_nazario
from parse_spamassassin import parse_spamassassin

RAW_DIR = os.path.join("data", "raw")
PROCESSED_FILE = os.path.join("data", "processed", "emails.csv")

def run_ingestion_pipeline():
    print("--- STARTING WEEK 1 DATA INGESTION PIPELINE ---")
    
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    
    df_enron_nazario = parse_enron_and_nazario(RAW_DIR)
    df_spamassassin = parse_spamassassin(RAW_DIR)
    
    combined_df = pd.concat([df_enron_nazario, df_spamassassin], ignore_index=True)
    
    if combined_df.empty:
        print("[Notice] No raw files found. Generating baseline dataset...")
        combined_df = pd.DataFrame({
            "text": [
                "URGENT: Your account access has been restricted. Click http://verify-bank.com immediately.",
                "Hey Team, please review the attached schedule for tomorrow's meeting.",
                "Dear customer, suspicious login detected. Reset password at http://fake-login.com",
                "Hi Anshika, please share the Week 1 project updates when free."
            ],
            "label": [1, 0, 1, 0]
        })
    
    print("Performing data cleaning and deduplication...")
    combined_df = combined_df.dropna().drop_duplicates(subset=["text"])
    
    combined_df.to_csv(PROCESSED_FILE, index=False)
    print(f"Pipeline Execution Complete! Output saved to: {PROCESSED_FILE}")
    print(f"Total Cleaned Samples: {len(combined_df)}")

if __name__ == "__main__":
    run_ingestion_pipeline()