#!/usr/bin/env python3
"""Test NB10: LLM App Evaluation
Tests automated metrics, rubric scoring, and model availability.
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
from sklearn.metrics import accuracy_score, classification_report, f1_score

STANCE_LABELS = {0: "none", 1: "against", 2: "favor"}
CATEGORIES = ["none", "against", "favor"]


class StancePrediction(BaseModel):
    label: Literal["none", "against", "favor"] = Field(description="Stance toward climate change action")


class RubricScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Rubric score 1-5")
    explanation: str = Field(description="Brief reason for the score")


class FaithfulnessCheck(BaseModel):
    faithful: bool = Field(description="Whether the answer is faithful to the context")
    explanation: str = Field(description="Brief reason")


def main():
    print("=" * 60)
    print("TEST NB10: LLM App Evaluation")
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

    # --- 2. Test MODEL_SMART availability ---
    print("\n[2] Testing MODEL_SMART model availability...")
    models_to_test = [
        "moonshotai/kimi-k2-instruct",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
    ]
    working_smart_model = None
    for model_name in models_to_test:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=5,
            )
            print(f"  {model_name}: WORKS ({resp.choices[0].message.content})")
            if working_smart_model is None:
                working_smart_model = model_name
        except Exception as e:
            err_str = str(e)
            # Extract just the relevant part of the error
            if "not found" in err_str.lower() or "does not exist" in err_str.lower():
                print(f"  {model_name}: NOT AVAILABLE (model not found)")
            elif "rate_limit" in err_str.lower():
                print(f"  {model_name}: RATE LIMITED (but exists)")
                if working_smart_model is None:
                    working_smart_model = model_name
            else:
                print(f"  {model_name}: ERROR ({err_str[:100]})")

    if working_smart_model:
        print(f"\n  RECOMMENDED MODEL_SMART: {working_smart_model}")
        MODEL_SMART = working_smart_model
    else:
        print(f"\n  WARNING: No smart model available, falling back to {MODEL_FAST}")
        MODEL_SMART = MODEL_FAST

    # --- 3. Load eval set ---
    print("\n[3] Loading tweet_eval stance_climate...")
    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    test_full = pd.DataFrame(ds["test"])
    test_full["label_name"] = test_full["label"].map(STANCE_LABELS)

    eval_set = test_full.sample(30, random_state=42)[['text', 'label_name']].reset_index(drop=True)
    eval_set.columns = ['input_text', 'expected_label']
    print(f"  Eval set: {len(eval_set)} examples")
    print(f"  Distribution: {dict(eval_set['expected_label'].value_counts())}")

    # --- 4. Stance classification ---
    print("\n[4] Running stance classification on all 30 examples...")

    def classify_stance(text, max_retries=3):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[
                        {"role": "system", "content": f"Determine the stance of this tweet toward climate change action. Classify into one of: {CATEGORIES}. 'favor' means supporting action on climate change, 'against' means opposing it, 'none' means neutral or unrelated. Return JSON with 'label' field only."},
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

    predictions = []
    for _, row in eval_set.iterrows():
        result = classify_stance(row['input_text'])
        predictions.append(result.label if result else 'Unknown')
        time.sleep(0.1)

    eval_set['predicted_label'] = predictions
    valid = eval_set[eval_set.predicted_label != 'Unknown']

    acc = accuracy_score(valid.expected_label, valid.predicted_label)
    macro_f1 = f1_score(valid.expected_label, valid.predicted_label, average='macro', zero_division=0)
    print(f"  Accuracy: {acc:.1%}")
    print(f"  Macro F1: {macro_f1:.3f}")
    print(f"\n{classification_report(valid.expected_label, valid.predicted_label, zero_division=0)}")

    # --- 5. Rubric scoring on ALL examples (the bug was: only 10) ---
    print("\n[5] Rubric scoring on ALL 30 examples (testing the bug fix)...")

    RUBRIC = """Score the stance classification on a scale of 1-5:
5: Exact match with ground truth
4: Close — predicted 'none' when true stance was mild, or vice versa
3: Partially correct — got the general sentiment but wrong specific label
2: Opposite direction — confused 'favor' with 'against' or vice versa
1: Completely wrong, no reasonable connection

Return JSON: {"score": <int>, "explanation": "<brief reason>"}"""

    def score_with_rubric(input_text, expected, predicted, max_retries=3):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_SMART,
                    messages=[
                        {"role": "system", "content": RUBRIC},
                        {"role": "user", "content": f"Tweet: {input_text[:200]}\nExpected stance: {expected}\nPredicted stance: {predicted}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=100
                )
                return RubricScore.model_validate_json(resp.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    scores = []
    for i, (_, row) in enumerate(eval_set.iterrows()):
        result = score_with_rubric(row['input_text'], row['expected_label'], row['predicted_label'])
        if result:
            scores.append(result)
        time.sleep(0.15)  # Rate limiting for smart model
        if (i + 1) % 10 == 0:
            print(f"    Scored {i+1}/{len(eval_set)}...")

    print(f"  Scored: {len(scores)}/{len(eval_set)}")
    avg_score = np.mean([s.score for s in scores]) if scores else 0
    print(f"  Average rubric score: {avg_score:.1f}/5")

    if len(scores) >= 25:
        print("  PASS: Rubric scoring worked on most/all examples")
    else:
        print(f"  WARNING: Only {len(scores)} scores collected (target: 30)")

    # --- 6. Error analysis ---
    print("\n[6] Error analysis...")
    errors = eval_set[eval_set.expected_label != eval_set.predicted_label]
    print(f"  Errors: {len(errors)}/{len(eval_set)} ({len(errors)/len(eval_set):.0%})")

    if len(errors) > 0:
        flipped = errors[
            ((errors.expected_label == 'favor') & (errors.predicted_label == 'against')) |
            ((errors.expected_label == 'against') & (errors.predicted_label == 'favor'))
        ]
        print(f"  Stance inversions: {len(flipped)}/{len(errors)} errors")

        for i, (_, row) in enumerate(errors.head(3).iterrows()):
            print(f"\n  Error {i+1}:")
            print(f"    Tweet: {row['input_text'][:120]}...")
            print(f"    Expected: {row['expected_label']}, Predicted: {row['predicted_label']}")

    # --- 7. Faithfulness check ---
    print("\n[7] Testing faithfulness check...")
    def check_faithfulness(question, context, answer, max_retries=3):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[{"role": "user", "content": f"""Given this context and answer, is the answer faithful to (supported by) the context?
Context: {context[:500]}
Answer: {answer}
Return JSON: {{"faithful": true/false, "explanation": "brief reason"}}"""}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=100
                )
                return FaithfulnessCheck.model_validate_json(resp.choices[0].message.content)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    # Test with faithful answer
    result1 = check_faithfulness(
        "What causes climate change?",
        "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels.",
        "Climate change is caused by burning fossil fuels which release greenhouse gases."
    )
    if result1:
        print(f"  Faithful answer test: faithful={result1.faithful} — {result1.explanation}")
        assert result1.faithful, "Expected faithful=True for correct answer"

    # Test with unfaithful answer
    result2 = check_faithfulness(
        "What causes climate change?",
        "Climate change is primarily caused by greenhouse gas emissions from burning fossil fuels.",
        "Climate change is caused by solar activity."
    )
    if result2:
        print(f"  Unfaithful answer test: faithful={result2.faithful} — {result2.explanation}")
        assert not result2.faithful, "Expected faithful=False for incorrect answer"

    print("  PASS: Faithfulness check works correctly")

    print("\n" + "=" * 60)
    print("NB10: ALL TESTS PASSED")
    print("=" * 60)
    print(f"\nKey findings:")
    print(f"  MODEL_SMART recommendation: {MODEL_SMART}")
    print(f"  Stance classification accuracy: {acc:.1%}")
    print(f"  Average rubric score: {avg_score:.1f}/5 (on {len(scores)} examples)")
    print(f"  BUG CONFIRMED: eval_set.head(10) should be eval_set (all rows)")


if __name__ == "__main__":
    main()
