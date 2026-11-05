import os
import pandas as pd
from header_features import extract_header_features
from content_features import get_tfidf_features, explore_distilbert_embeddings

PROCESSED_FILE = os.path.join("data", "processed", "emails.csv")
OUTPUT_FEATURE_FILE = os.path.join("data", "processed", "features.csv")

def run_feature_extraction():
    print("--- STARTING WEEK 2 FEATURE EXTRACTION PIPELINE ---")
    
    if not os.path.exists(PROCESSED_FILE):
        print(f"Error: File '{PROCESSED_FILE}' not found! Run Week 1 script first.")
        return
        
    df = pd.read_csv(PROCESSED_FILE)
    
    print("Extracting Header & Security features (SPF, DKIM, DMARC, Typosquatting)...")
    header_feats = [extract_header_features(text) for text in df['text']]
    header_df = pd.DataFrame(header_feats)
    
    print("Extracting TF-IDF Content features...")
    tfidf_df = get_tfidf_features(df['text'].tolist(), max_features=10)
    
    print("Exploring DistilBERT embedding metrics...")
    distil_df = explore_distilbert_embeddings(df['text'].tolist())
    
    # Combine all extracted feature matrices
    final_features_df = pd.concat([df[['label']], header_df, distil_df, tfidf_df], axis=1)
    
    # Save the feature output
    final_features_df.to_csv(OUTPUT_FEATURE_FILE, index=False)
    print(f"Week 2 Complete! Feature matrix saved to: {OUTPUT_FEATURE_FILE}")
    print(f"Total Features Extracted: {len(final_features_df.columns) - 1}")

if __name__ == "__main__":
    run_feature_extraction()