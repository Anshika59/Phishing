import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(text_list, max_features=10):
    """
    Module written by Riya: Implements TF-IDF vectorization for email content.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(text_list)
    feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

def explore_distilbert_embeddings(text_list):
    """
    Module written by Riya: Basic exploration script for DistilBERT text metrics.
    """
    word_counts = []
    char_counts = []
    
    for text in text_list:
        words = str(text).split()
        word_counts.append(len(words))
        char_counts.append(sum(len(w) for w in words))
        
    return pd.DataFrame({
        "bert_word_count": word_counts,
        "bert_char_count": char_counts
    })