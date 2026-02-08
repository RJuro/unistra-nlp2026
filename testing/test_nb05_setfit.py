"""Test script for NB05: SetFit Few-shot Classification"""
import sys, os, time
start = time.time()

print("=" * 60)
print("TEST: NB05 — SetFit Few-shot Classification")
print("=" * 60)

# ── 1. Load dataset ──
print("\n[1/7] Loading environmental claims dataset...")
from datasets import load_dataset
import pandas as pd
import numpy as np

try:
    dataset = load_dataset("climatebert/environmental_claims")
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"  Labels: {dict(train_df['label'].value_counts())}")
    print("  ✓ Dataset OK")
except Exception as e:
    print(f"  ✗ Failed to load dataset: {e}")
    sys.exit(1)

# ── 2. Sample few-shot set ──
print("\n[2/7] Sampling few-shot sets...")
def sample_few_shot(df, n_per_class=4, seed=42):
    np.random.seed(seed)
    samples = []
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        chosen = label_df.sample(n=min(n_per_class, len(label_df)), random_state=seed)
        samples.append(chosen)
    return pd.concat(samples).reset_index(drop=True)

few_shot_8 = sample_few_shot(train_df, n_per_class=4, seed=42)
few_shot_16 = sample_few_shot(train_df, n_per_class=8, seed=42)
print(f"  8-shot: {len(few_shot_8)} examples, labels: {dict(few_shot_8['label'].value_counts())}")
print(f"  16-shot: {len(few_shot_16)} examples")
assert len(few_shot_8) == 8, f"Expected 8, got {len(few_shot_8)}"
print("  ✓ Sampling OK")

# ── 3. E5 prefix handling ──
print("\n[3/7] Testing E5 prefix formatting...")
MODEL_NAME = "intfloat/e5-small"

def format_texts_for_model(texts, model_name):
    if "e5" in model_name.lower():
        return ["query: " + t for t in texts]
    return list(texts)

test_texts = ["This is a test", "Another test"]
formatted = format_texts_for_model(test_texts, MODEL_NAME)
assert formatted[0].startswith("query: "), "E5 prefix missing"
print(f"  Formatted: '{formatted[0][:30]}...'")
print("  ✓ Prefix OK")

# ── 4. Train SetFit on 8-shot ──
print("\n[4/7] Training SetFit (8-shot)...")
import warnings
warnings.filterwarnings("ignore")

from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset

# Prepare data
train_texts = format_texts_for_model(few_shot_8['text'].tolist(), MODEL_NAME)
train_labels = few_shot_8['label'].tolist()
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})

test_texts = format_texts_for_model(test_df['text'].tolist(), MODEL_NAME)
test_labels = test_df['label'].tolist()
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

model = SetFitModel.from_pretrained(MODEL_NAME)
args = TrainingArguments(
    batch_size=16,
    num_iterations=20,
    num_epochs=4,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
preds = model.predict(test_texts)
acc_8shot = accuracy_score(test_labels, preds)
f1_8shot = f1_score(test_labels, preds, average='macro')
print(f"  8-shot Accuracy: {acc_8shot:.4f}")
print(f"  8-shot Macro-F1: {f1_8shot:.4f}")
assert 0.40 < acc_8shot < 0.95, f"Accuracy {acc_8shot} outside expected range"
print("  ✓ SetFit 8-shot training OK")

# ── 5. TF-IDF baseline ──
print("\n[5/7] TF-IDF baseline...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_tfidf = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
pipe_tfidf.fit(few_shot_8['text'].tolist(), few_shot_8['label'].tolist())
tfidf_preds = pipe_tfidf.predict(test_df['text'].tolist())
acc_tfidf = accuracy_score(test_labels, tfidf_preds)
f1_tfidf = f1_score(test_labels, tfidf_preds, average='macro')
print(f"  TF-IDF Accuracy: {acc_tfidf:.4f}")
print(f"  TF-IDF Macro-F1: {f1_tfidf:.4f}")

print(f"\n  SetFit vs TF-IDF (8-shot):")
print(f"    SetFit: acc={acc_8shot:.4f}, F1={f1_8shot:.4f}")
print(f"    TF-IDF: acc={acc_tfidf:.4f}, F1={f1_tfidf:.4f}")
if acc_8shot > acc_tfidf:
    print("    ✓ SetFit beats TF-IDF (as expected with few examples)")
else:
    print("    ⚠ TF-IDF beat SetFit — unusual with only 8 examples")

# ── 6. LLM bootstrapping test (optional) ──
print("\n[6/7] LLM bootstrapping test...")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

llm_bootstrap_ok = False
if GROQ_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    LLM_MODEL = "openai/gpt-oss-20b"  # NB05 uses this for paraphrases

    test_text = few_shot_8.iloc[0]['text']
    test_label = few_shot_8.iloc[0]['label']
    label_desc = {0: "not an environmental claim", 1: "an environmental claim"}

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": f"Generate 2 paraphrases of this text that are clearly {label_desc[test_label]}. Return as JSON list of strings."},
                {"role": "user", "content": test_text[:500]},
            ],
            temperature=0.9, max_tokens=300,
        )
        content = resp.choices[0].message.content
        print(f"  LLM response preview: {content[:100]}...")
        # Try to parse as JSON list
        import json
        try:
            paraphrases = json.loads(content)
            if isinstance(paraphrases, list) and len(paraphrases) >= 1:
                print(f"  Generated {len(paraphrases)} paraphrases")
                llm_bootstrap_ok = True
                print("  ✓ LLM bootstrapping OK")
            else:
                print(f"  ⚠ Unexpected format: {type(paraphrases)}")
        except json.JSONDecodeError:
            print(f"  ⚠ Could not parse JSON from LLM output")
    except Exception as e:
        print(f"  ⚠ LLM call failed: {e}")
else:
    print("  ⚠ No GROQ_API_KEY — skipping LLM bootstrapping")

# ── 7. Classification report ──
print("\n[7/7] Full classification report (8-shot SetFit)...")
print(classification_report(test_labels, preds, target_names=["No claim", "Env. claim"], zero_division=0))

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"NB05 RESULT: PASS")
print(f"  SetFit 8-shot: acc={acc_8shot:.4f}, F1={f1_8shot:.4f}")
print(f"  TF-IDF 8-shot: acc={acc_tfidf:.4f}, F1={f1_tfidf:.4f}")
print(f"  LLM bootstrap: {'OK' if llm_bootstrap_ok else 'skipped/failed'}")
print(f"  Model: {MODEL_NAME}")
print(f"  LLM: openai/gpt-oss-20b (for paraphrases)")
print(f"  Time: {elapsed:.1f}s")
print(f"{'=' * 60}")
