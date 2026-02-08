#!/usr/bin/env python3
"""Test NB09: Fine-tuning Qwen3 — DATA PREP ONLY (no CUDA)
Tests data loading, both paths (CSV and HF fallback), and chat template formatting.
"""

import os
import sys
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CATEGORIES = sorted(EMOTION_LABELS)


def main():
    print("=" * 60)
    print("TEST NB09: Fine-tuning — Data Prep Only (no CUDA)")
    print("=" * 60)

    # --- 1. Test HF data loading ---
    print("\n[1] Loading dair-ai/emotion from HuggingFace...")
    emotion_ds = load_dataset("dair-ai/emotion")
    train_full = pd.DataFrame(emotion_ds["train"])
    train_full["label_name"] = train_full["label"].map(lambda x: EMOTION_LABELS[x])
    test_full = pd.DataFrame(emotion_ds["test"])
    test_full["label_name"] = test_full["label"].map(lambda x: EMOTION_LABELS[x])

    train_subset = train_full.sample(800, random_state=42)
    eval_subset = test_full.sample(200, random_state=42)

    print(f"  Train: {len(train_subset)}, Eval: {len(eval_subset)}")
    print(f"  Train distribution: {dict(train_subset['label_name'].value_counts())}")
    print("  PASS: HF loading works")

    # --- 2. Test CSV path ---
    print("\n[2] Testing CSV loading path...")
    csv_path = os.path.join(os.path.dirname(__file__), "emotion_distilled_labels.csv")

    if os.path.exists(csv_path):
        distilled = pd.read_csv(csv_path)
        distilled = distilled.rename(columns={"llm_label": "label_name"})
        csv_train = distilled.sample(min(800, len(distilled)), random_state=42)
        print(f"  CSV loaded: {len(distilled)} rows, using {len(csv_train)} for training")
        print(f"  CSV distribution: {dict(csv_train['label_name'].value_counts())}")
        print("  PASS: CSV path works")
    else:
        print(f"  CSV not found at {csv_path} (expected if NB08 test hasn't run)")
        print("  OK: HF fallback would be used")

    # --- 3. Test chat template formatting ---
    print("\n[3] Testing chat template formatting...")
    # Use transformers tokenizer instead of unsloth (no CUDA needed)
    from transformers import AutoTokenizer

    # Use a small tokenizer for template testing
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        print(f"  Using Qwen2.5-0.5B-Instruct tokenizer (same chat template family)")
    except Exception as e:
        print(f"  WARNING: Could not load Qwen tokenizer: {e}")
        print("  Skipping chat template test")
        return

    def format_instruction(row):
        return {
            "text": tokenizer.apply_chat_template([
                {"role": "system", "content": f"You classify tweets into one of these emotion categories: {CATEGORIES}. Respond with only the emotion label."},
                {"role": "user", "content": f"Classify the emotion in this tweet:\n\n{row['text'][:500]}"},
                {"role": "assistant", "content": row['label_name']}
            ], tokenize=False)
        }

    sample_rows = train_subset.head(5)
    formatted = [format_instruction(row) for _, row in sample_rows.iterrows()]

    print(f"\n  Sample formatted example:")
    print(f"  {formatted[0]['text'][:400]}...")
    print(f"\n  All 5 samples formatted successfully")

    # Verify format contains expected elements
    for i, fmt in enumerate(formatted):
        text = fmt["text"]
        assert "emotion categories" in text, f"Sample {i}: missing system prompt"
        assert "Classify the emotion" in text, f"Sample {i}: missing user prompt"
        label = sample_rows.iloc[i]['label_name']
        assert label in text, f"Sample {i}: missing label '{label}'"
    print("  PASS: Chat template produces valid format")

    # --- 4. Test Dataset creation ---
    print("\n[4] Testing HuggingFace Dataset creation...")
    train_data = [format_instruction(row) for _, row in train_subset.head(50).iterrows()]
    train_dataset = Dataset.from_list(train_data)
    print(f"  Created Dataset with {len(train_dataset)} examples")
    print(f"  Columns: {train_dataset.column_names}")
    assert "text" in train_dataset.column_names
    print("  PASS: Dataset creation works")

    # --- 5. Verify both paths produce valid data ---
    print("\n[5] Verifying data quality...")
    labels_in_data = set(train_subset['label_name'].unique())
    expected_labels = set(EMOTION_LABELS)
    missing = expected_labels - labels_in_data
    if missing:
        print(f"  WARNING: Missing labels in 800-sample: {missing}")
    else:
        print(f"  All 6 emotion labels present in training data")

    # Check for empty texts
    empty = train_subset[train_subset['text'].str.strip() == '']
    print(f"  Empty texts: {len(empty)}")
    assert len(empty) == 0, "Found empty texts in training data"
    print("  PASS: Data quality checks passed")

    print("\n" + "=" * 60)
    print("NB09: ALL DATA PREP TESTS PASSED")
    print("=" * 60)
    print("\nNote: Training requires CUDA (unsloth). Only data prep was tested locally.")


if __name__ == "__main__":
    main()
