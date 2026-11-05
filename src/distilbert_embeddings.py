import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel

# Load lightweight DistilBERT tokenizer and model
TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MODEL = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_distilbert_embeddings(text_list, max_len=128):
    """
    Fine-tuned embedding extractor using DistilBERT mean pooling 
    for context-aware phishing feature representation.
    """
    MODEL.eval()
    encoded = TOKENIZER(
        text_list, 
        padding=True, 
        truncation=True, 
        max_length=max_len, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = MODEL(**encoded)
        # Mean pooling across tokens to get single sentence embedding vector
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
    return embeddings