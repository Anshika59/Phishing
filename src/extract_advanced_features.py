import os
import pandas as pd
from url_attachment_features import extract_url_features, check_risky_attachments
from distilbert_embeddings import get_distilbert_embeddings

def find_column(df, possible_names):
    for col in possible_names:
        if col in df.columns:
            return col
    return None

def build_advanced_feature_pipeline():
    processed_data_path = os.path.join("data", "processed", "emails.csv")
    output_path = os.path.join("data", "processed", "features_v2.csv")
    
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Missing {processed_data_path}. Run data ingestion pipeline first.")
        
    df = pd.read_csv(processed_data_path)
    
    # Auto-detect text column name
    body_col = find_column(df, ["body", "text", "content", "message", "email_text"])
    if not body_col:
        df["body"] = ""
        body_col = "body"
        
    # Auto-detect attachments column name
    attach_col = find_column(df, ["attachments", "attachment", "files"])
    if not attach_col:
        df["attachments"] = ""
        attach_col = "attachments"

    # Auto-detect label column name
    label_col = find_column(df, ["label", "target", "class", "is_phishing"])

    print(f"Using '{body_col}' column for text feature extraction...")
    print("Extracting URL and Attachment Features...")
    
    url_feats = df[body_col].astype(str).fillna("").apply(extract_url_features)
    url_df = pd.DataFrame(list(url_feats))
    
    attachment_flags = df[attach_col].astype(str).fillna("").apply(check_risky_attachments)
    url_df["has_risky_attachment"] = attachment_flags
    
    print("Generating DistilBERT embeddings...")
    bodies = df[body_col].astype(str).fillna("").tolist()
    
    # Process text batch for embeddings
    embeddings = get_distilbert_embeddings(bodies[:100])
    emb_cols = [f"distilbert_emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols)
    
    label_series = df[label_col] if label_col else pd.Series([0] * len(df), name="label")
    
    # Combine features into final dataset
    features_df = pd.concat([label_series, url_df, emb_df], axis=1).fillna(0)
    features_df.to_csv(output_path, index=False)
    print(f"Pipeline complete! Advanced features saved to: {output_path}")

if __name__ == "__main__":
    build_advanced_feature_pipeline()