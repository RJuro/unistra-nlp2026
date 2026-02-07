"""Data loading utilities for UNISTRA NLP 2026 workshop notebooks."""
import pandas as pd
import re

# GitHub raw content base URL
REPO_BASE = "https://raw.githubusercontent.com/RJuro/unistra-nlp2026/main"

def load_dk_posts(source="github"):
    """Load the dk_posts demo dataset.

    Args:
        source: "github" (default) or a local file path
    """
    if source == "github":
        url = f"{REPO_BASE}/data/dk_posts_synth_en_processed.json"
        df = pd.read_json(url, orient='records')
    else:
        df = pd.read_json(source, orient='records')

    df['text'] = df['title'] + " . " + df['selftext']
    df['text_clean'] = df['text'].apply(clean_text)
    return df


def clean_text(text):
    """Basic text cleaning: lowercase, strip, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def get_train_test_split(df, test_size=0.25, random_state=42):
    """Get a stratified train/test split of the dk_posts dataset."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        df['text_clean'], df['label'],
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
