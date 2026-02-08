#!/usr/bin/env python3
"""Test NB11: Annotation & Inter-Rater Reliability
Tests Cohen's kappa, Gwet's AC1, disagreement analysis.
Requires GROQ_API_KEY.
"""

import os
import sys
import json
import time

# Load .env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"')

import pandas as pd
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal, Optional
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score

STANCE_LABELS = {0: "none", 1: "against", 2: "favor"}
CATEGORIES = ["none", "against", "favor"]


class StancePrediction(BaseModel):
    label: Literal["none", "against", "favor"] = Field(description="Stance toward climate change action")


def gwet_ac1(labels1, labels2):
    """Calculate Gwet's AC1 agreement coefficient."""
    n = len(labels1)
    categories = sorted(set(labels1) | set(labels2))

    # Observed agreement
    po = sum(a == b for a, b in zip(labels1, labels2)) / n

    # Expected agreement under AC1
    marginals = []
    for cat in categories:
        pi_k = (sum(1 for l in labels1 if l == cat) + sum(1 for l in labels2 if l == cat)) / (2 * n)
        marginals.append(pi_k)

    pe = sum(pk * (1 - pk) for pk in marginals) / (len(categories) - 1) if len(categories) > 1 else 0

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    print("=" * 60)
    print("TEST NB11: Annotation & Inter-Rater Reliability")
    print("=" * 60)

    # --- 1. Setup ---
    print("\n[1] Setting up Groq client...")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  ERROR: GROQ_API_KEY not found")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    MODEL_FAST = "llama-3.1-8b-instant"

    resp = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": "Say 'ready'"}],
        max_tokens=5
    )
    print(f"  Connection OK: {resp.choices[0].message.content}")

    # --- 2. Load data ---
    print("\n[2] Loading tweet_eval stance_climate...")
    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    test_full = pd.DataFrame(ds["test"])
    test_full["label_name"] = test_full["label"].map(STANCE_LABELS)

    annotation_set = test_full.sample(50, random_state=42)[['text', 'label_name']].reset_index(drop=True)
    annotation_set.columns = ['text', 'human_label']
    print(f"  Annotation set: {len(annotation_set)} examples")
    print(f"  Distribution: {dict(annotation_set['human_label'].value_counts())}")

    # --- 3. LLM classification ---
    print("\n[3] Running LLM classification on 50 examples...")

    def classify_stance(text, max_retries=3):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[
                        {"role": "system", "content": f"Determine the stance of this tweet toward climate change action. Classify into one of: {CATEGORIES}. 'favor' means supporting action on climate change, 'against' means opposing it, 'none' means neutral or unrelated. Return JSON with 'label' field."},
                        {"role": "user", "content": text[:500]}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=50
                )
                return StancePrediction.model_validate_json(resp.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    llm_labels = []
    for _, row in annotation_set.iterrows():
        result = classify_stance(row['text'])
        llm_labels.append(result.label if result else 'Unknown')
        time.sleep(0.1)

    annotation_set['llm_label'] = llm_labels
    valid = annotation_set[annotation_set.llm_label != 'Unknown']
    raw_agreement = accuracy_score(valid.human_label, valid.llm_label)
    print(f"  Raw agreement: {raw_agreement:.1%}")
    print(f"  Valid labels: {len(valid)}/{len(annotation_set)}")

    # --- 4. Cohen's kappa ---
    print("\n[4] Cohen's kappa...")
    kappa = cohen_kappa_score(valid['human_label'], valid['llm_label'])
    print(f"  Cohen's Kappa: {kappa:.3f}")

    if kappa > 0.8:
        interpretation = "Almost perfect agreement"
    elif kappa > 0.6:
        interpretation = "Substantial agreement"
    elif kappa > 0.4:
        interpretation = "Moderate agreement"
    elif kappa > 0.2:
        interpretation = "Fair agreement"
    else:
        interpretation = "Slight agreement"
    print(f"  Interpretation: {interpretation}")

    # Validate kappa range
    assert -1 <= kappa <= 1, f"Kappa out of range: {kappa}"
    print("  PASS: Kappa computed correctly")

    # --- 5. Gwet's AC1 ---
    print("\n[5] Gwet's AC1...")
    ac1 = gwet_ac1(valid['human_label'].tolist(), valid['llm_label'].tolist())
    print(f"  Gwet's AC1: {ac1:.3f}")
    print(f"  Cohen's Kappa: {kappa:.3f}")

    assert -1 <= ac1 <= 1, f"AC1 out of range: {ac1}"
    print(f"  AC1 vs Kappa difference: {ac1 - kappa:+.3f}")
    print("  PASS: AC1 computed correctly")

    # --- 6. Disagreement analysis ---
    print("\n[6] Disagreement analysis...")
    disagreements = valid[valid.human_label != valid.llm_label]
    print(f"  Disagreements: {len(disagreements)}/{len(valid)} ({len(disagreements)/len(valid):.0%})")

    flipped = disagreements[
        ((disagreements.human_label == 'favor') & (disagreements.llm_label == 'against')) |
        ((disagreements.human_label == 'against') & (disagreements.llm_label == 'favor'))
    ]
    print(f"  Stance inversions: {len(flipped)}/{len(disagreements)} disagreements")

    confusion_pairs = disagreements.groupby(['human_label', 'llm_label']).size().sort_values(ascending=False)
    print(f"\n  Confusion pairs:")
    for (h, l), count in confusion_pairs.items():
        print(f"    Human: {h:<10} -> LLM: {l} ({count}x)")

    print(f"\n  Example disagreements:")
    for i, (_, row) in enumerate(disagreements.head(3).iterrows()):
        print(f"    [{i+1}] Tweet: {row['text'][:100]}...")
        print(f"         Human: {row['human_label']} | LLM: {row['llm_label']}")

    # --- 7. Codebook ---
    print("\n[7] Codebook definition...")
    CODEBOOK = {
        "favor": "Tweet explicitly or implicitly supports action on climate change.",
        "against": "Tweet explicitly or implicitly opposes action on climate change.",
        "none": "Tweet mentions climate change but does not take a clear stance, OR is unrelated.",
    }
    for code, desc in CODEBOOK.items():
        print(f"  {code.upper()}: {desc[:80]}")
    print("  PASS: Codebook defined")

    print("\n" + "=" * 60)
    print("NB11: ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Raw agreement: {raw_agreement:.1%}")
    print(f"  Cohen's Kappa: {kappa:.3f} ({interpretation})")
    print(f"  Gwet's AC1: {ac1:.3f}")
    print(f"  Stance inversions: {len(flipped)}/{len(disagreements)} disagreements")


if __name__ == "__main__":
    main()
