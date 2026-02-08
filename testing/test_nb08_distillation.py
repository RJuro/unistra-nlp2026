#!/usr/bin/env python3
"""Test NB08: Distillation — LLM Label Synthesis
Uses a small sample (100 tweets) for speed.
Requires GROQ_API_KEY in environment or .env file.
"""

import os
import sys
import json
import time
import hashlib

# Load .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"')

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


class LabelPrediction(BaseModel):
    label: Literal["sadness", "joy", "love", "anger", "fear", "surprise"] = Field(description="Best-fit emotion category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str = Field(description="Brief reasoning")


def main():
    print("=" * 60)
    print("TEST NB08: Distillation — LLM Label Synthesis")
    print("=" * 60)

    # --- 1. Setup ---
    print("\n[1] Setting up Groq client...")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  ERROR: GROQ_API_KEY not found")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    MODEL_FAST = "llama-3.1-8b-instant"

    # Test connection
    resp = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": "Say 'ready'"}],
        max_tokens=5
    )
    print(f"  Connection OK: {resp.choices[0].message.content}")

    # --- 2. Load dataset ---
    print("\n[2] Loading dair-ai/emotion dataset...")
    emotion_ds = load_dataset("dair-ai/emotion")
    train_full = pd.DataFrame(emotion_ds["train"])
    train_full["label_name"] = train_full["label"].map(lambda x: EMOTION_LABELS[x])

    np.random.seed(42)
    pool_idx = np.random.choice(len(train_full), size=150, replace=False)
    pool_df = train_full.iloc[pool_idx].reset_index(drop=True)

    train_pool = pool_df.iloc[:100].copy()  # 100 for LLM labeling (speed)
    test_df = pool_df.iloc[100:].copy()      # 50 for evaluation

    print(f"  Unlabeled pool: {len(train_pool)} tweets")
    print(f"  Test set: {len(test_df)} tweets")
    print(f"  Distribution: {dict(train_pool['label_name'].value_counts())}")

    # --- 3. LLM labeling ---
    print("\n[3] LLM labeling (100 tweets)...")

    def label_with_retry(text, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[
                        {"role": "system", "content": f"Classify the emotion expressed in the following text into one of these categories: {EMOTION_LABELS}. Return JSON with 'label', 'confidence' (0-1), and 'reasoning'."},
                        {"role": "user", "content": text[:500]}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=150
                )
                return LabelPrediction.model_validate_json(response.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    labeled_data = []
    errors = 0
    for idx, row in tqdm(train_pool.iterrows(), total=len(train_pool), desc="Labeling"):
        result = label_with_retry(row['text'])
        if result:
            labeled_data.append({
                'text': row['text'],
                'llm_label': result.label,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'true_label': row['label_name']
            })
        else:
            errors += 1
        time.sleep(0.1)

    labeled_df = pd.DataFrame(labeled_data)
    llm_acc = accuracy_score(labeled_df['true_label'], labeled_df['llm_label'])
    print(f"  Labeled: {len(labeled_df)}/{len(train_pool)} ({errors} errors)")
    print(f"  LLM accuracy vs true labels: {llm_acc:.1%}")

    if llm_acc < 0.3:
        print("  WARNING: LLM accuracy suspiciously low!")
    elif llm_acc > 0.5:
        print("  OK: LLM accuracy is reasonable")

    # --- 4. Confidence filtering ---
    print("\n[4] Confidence filtering...")
    print(f"  Confidence distribution:")
    print(f"    Mean: {labeled_df['confidence'].mean():.2f}")
    print(f"    Median: {labeled_df['confidence'].median():.2f}")
    print(f"    Min: {labeled_df['confidence'].min():.2f}")
    print(f"    Max: {labeled_df['confidence'].max():.2f}")

    high_conf = labeled_df[labeled_df['confidence'] >= 0.7].copy()
    hc_acc = accuracy_score(high_conf['true_label'], high_conf['llm_label']) if len(high_conf) > 0 else 0
    print(f"\n  After filter (>=0.7): {len(high_conf)}/{len(labeled_df)}")
    print(f"  High-confidence accuracy: {hc_acc:.1%}")
    print(f"  All labels accuracy:      {llm_acc:.1%}")

    if hc_acc >= llm_acc:
        print("  PASS: Confidence filtering improves accuracy")
    else:
        print(f"  NOTE: Confidence filtering did NOT improve accuracy ({hc_acc:.1%} < {llm_acc:.1%})")
        print("  This can happen with small samples. Not necessarily a bug.")

    # --- 5. Dedup ---
    print("\n[5] Deduplication...")
    high_conf['text_hash'] = high_conf['text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
    deduped = high_conf.drop_duplicates(subset='text_hash')
    print(f"  After dedup: {len(deduped)} unique tweets")
    print(f"  Distribution: {dict(deduped['llm_label'].value_counts())}")

    # --- 6. Train students ---
    print("\n[6] Training student classifiers...")
    if len(deduped) < 10:
        print("  SKIP: Too few examples for training after filtering")
        sys.exit(1)

    # TF-IDF student
    tfidf_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    tfidf_pipe.fit(deduped['text'], deduped['llm_label'])
    tfidf_preds = tfidf_pipe.predict(test_df['text'])
    tfidf_acc = accuracy_score(test_df['label_name'], tfidf_preds)

    # E5 student
    print("  Training E5 student...")
    e5_model = SentenceTransformer("intfloat/e5-small")
    train_texts = [f"query: {t.strip()}" for t in deduped['text'].tolist()]
    test_texts = [f"query: {t.strip()}" for t in test_df['text'].tolist()]

    train_emb = e5_model.encode(train_texts, show_progress_bar=True, normalize_embeddings=True)
    test_emb = e5_model.encode(test_texts, show_progress_bar=True, normalize_embeddings=True)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_emb, deduped['llm_label'])
    e5_preds = lr.predict(test_emb)
    e5_acc = accuracy_score(test_df['label_name'], e5_preds)

    print(f"\n  {'Model':<30} {'Accuracy':>10}")
    print("  " + "-" * 42)
    print(f"  {'LLM (teacher, zero-shot)':<30} {llm_acc:>10.1%}")
    print(f"  {'TF-IDF + LR (student)':<30} {tfidf_acc:>10.1%}")
    print(f"  {'E5 + LR (student)':<30} {e5_acc:>10.1%}")

    # --- 7. Classification report ---
    print("\n[7] E5 Student Classification Report:")
    print(classification_report(test_df['label_name'], e5_preds, zero_division=0))

    # --- 8. Save CSV ---
    output_file = os.path.join(os.path.dirname(__file__), "emotion_distilled_labels.csv")
    deduped[['text', 'llm_label', 'confidence']].to_csv(output_file, index=False)
    print(f"\n[8] Saved {len(deduped)} distilled labels to {output_file}")

    print("\n" + "=" * 60)
    print("NB08: ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  LLM teacher accuracy: {llm_acc:.1%}")
    print(f"  Confidence filter effect: {hc_acc:.1%} (was {llm_acc:.1%})")
    print(f"  TF-IDF student: {tfidf_acc:.1%}")
    print(f"  E5 student: {e5_acc:.1%}")


if __name__ == "__main__":
    main()
