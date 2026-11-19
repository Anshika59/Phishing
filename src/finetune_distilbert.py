import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class PhishingEmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def load_email_texts():
    """
    Loads cleaned email text and labels from data/processed/emails.csv.
    """
    emails_path = os.path.join("data", "processed", "emails.csv")
    if not os.path.exists(emails_path):
        raise FileNotFoundError(f"Missing {emails_path}. Run Week 1-2 data ingestion first.")

    df = pd.read_csv(emails_path)

    # Extract text column flexibly
    if "text" in df.columns:
        texts = df["text"].fillna("").astype(str).tolist()
    elif "subject" in df.columns and "body" in df.columns:
        texts = (df["subject"].fillna("") + " " + df["body"].fillna("")).astype(str).tolist()
    elif "body" in df.columns:
        texts = df["body"].fillna("").astype(str).tolist()
    else:
        text_cols = [c for c in df.columns if c != "label"]
        texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    labels = df["label"].astype(int).tolist()
    return texts, labels


def finetune_distilbert_model(epochs=2, batch_size=4, lr=2e-5):
    """
    Fine-tunes distilbert-base-uncased on the phishing dataset and saves the artifact.
    """
    print("\nInitializing DistilBERT Fine-Tuning Pipeline...")
    texts, labels = load_email_texts()

    # Safe train/validation split for small or large datasets
    label_series = pd.Series(labels)
    min_class = label_series.value_counts().min()
    test_count = max(1, int(len(labels) * 0.2))
    stratify_labels = labels if (min_class >= 2 and test_count >= 2) else None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=stratify_labels
    )

    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_dataset = PhishingEmailDataset(train_encodings, train_labels)
    val_dataset = PhishingEmailDataset(val_encodings, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"--> Fine-tuning DistilBERT on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"    Epoch {epoch + 1}/{epochs} | Training Loss: {avg_loss:.4f}")

    # Evaluate on validation set
    model.eval()
    preds, true_vals = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            true_vals.extend(batch["labels"].numpy())

    val_acc = accuracy_score(true_vals, preds)
    val_f1 = f1_score(true_vals, preds, zero_division=0)

    # Save fine-tuned transformer model
    save_dir = os.path.join("models", "distilbert_phishing")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"--> Validation Accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}")
    print(f"--> Fine-tuned DistilBERT saved to {save_dir}\n")

    return {"Val_Accuracy": val_acc, "Val_F1": val_f1}


if __name__ == "__main__":
    finetune_distilbert_model()