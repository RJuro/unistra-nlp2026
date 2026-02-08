"""Test script for NB03b: LLM Zero-shot (Local Ollama)"""
import sys, os, time, json, re
start = time.time()

print("=" * 60)
print("TEST: NB03b — LLM Zero-shot (Local Ollama)")
print("=" * 60)

# ── 1. Ollama connectivity ──
print("\n[1/7] Checking Ollama server...")
import requests

MODEL_NAME = "ministral-3:8b"

try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m['name'] for m in resp.json().get('models', [])]
    print(f"  Ollama running. Models: {models}")
    if not any(MODEL_NAME.split(':')[0] in m for m in models):
        print(f"  ⚠ {MODEL_NAME} not found — pulling...")
        os.system(f"ollama pull {MODEL_NAME}")
except Exception as e:
    print(f"  ✗ Ollama not running: {e}")
    sys.exit(1)

# ── 2. Quick test ──
print("\n[2/7] Quick inference test...")
import ollama

try:
    test_resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one word."}],
        stream=False
    )
    print(f"  Response: {test_resp['message']['content'].strip()[:50]}")
    print("  ✓ Ollama inference OK")
except Exception as e:
    print(f"  ✗ Ollama inference failed: {e}")
    sys.exit(1)

# ── 3. Data loading ──
print("\n[3/7] Loading dataset...")
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

DATA_URL = "https://raw.githubusercontent.com/RJuro/unistra-nlp2026/main/data/dk_posts_synth_en_processed.json"
df = pd.read_json(DATA_URL)
df['text'] = df['title'] + ' . ' + df['selftext']
df['text_clean'] = df['text'].apply(lambda t: re.sub(r'\s+', ' ', t.lower()).strip())
print(f"  Loaded {len(df)} rows")

# ── 4. Schema definition + constrained decoding ──
print("\n[4/7] Schema-enforced classification...")
CATEGORIES = [
    "Love & Dating", "Family Dynamics", "Work, Study & Career",
    "Friendship & Social Life", "Health & Wellness (Physical and Mental)",
    "Personal Finance & Housing", "Practical Questions & Everyday Life",
    "Everyday Observations & Rants"
]

class SingleLabelPrediction(BaseModel):
    predicted_label: Literal[
        "Love & Dating", "Family Dynamics", "Work, Study & Career",
        "Friendship & Social Life", "Health & Wellness (Physical and Mental)",
        "Personal Finance & Housing", "Practical Questions & Everyday Life",
        "Everyday Observations & Rants"
    ] = Field(description="The single best-fit category for this post.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")

# Test constrained decoding with format= parameter
def classify_post(text):
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"Classify this text into exactly one of these categories: {CATEGORIES}. Return predicted_label and confidence (0-1)."},
                {"role": "user", "content": text[:500]}
            ],
            format=SingleLabelPrediction.model_json_schema(),
            stream=False
        )
        content = response['message']['content']
        parsed = SingleLabelPrediction.model_validate_json(content)
        return parsed
    except Exception as e:
        print(f"    Error: {e}")
        return None

# Test single post first
print("  Testing single post...")
test_result = classify_post(df.iloc[0]['text_clean'])
if test_result:
    print(f"  Result: {test_result.predicted_label} (conf: {test_result.confidence:.2f})")
    print(f"  True:   {df.iloc[0]['label']}")
    print("  ✓ Schema enforcement works!")
else:
    print("  ✗ Schema enforcement failed")
    sys.exit(1)

# ── 5. Batch classification (15 posts for speed) ──
print("\n[5/7] Batch classification (15 posts)...")
sample = df.head(15).copy()
results = []
for i, row in sample.iterrows():
    result = classify_post(row['text_clean'])
    if result:
        results.append({
            'true_label': row['label'],
            'predicted_label': result.predicted_label,
            'confidence': result.confidence,
        })
    # No sleep needed for local Ollama

valid = pd.DataFrame(results)
n_success = len(valid)
print(f"  Successful: {n_success}/15")

from sklearn.metrics import accuracy_score, classification_report
if n_success >= 5:
    acc = accuracy_score(valid['true_label'], valid['predicted_label'])
    print(f"  Accuracy: {acc:.4f}")
    avg_conf = valid['confidence'].mean()
    print(f"  Avg confidence: {avg_conf:.3f}")
else:
    acc = 0.0
    print("  ⚠ Too few successful predictions")

# ── 6. Few-shot with integer IDs ──
print("\n[6/7] Few-shot classification (integer ID schema)...")
CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}
ID_TO_CATEGORY = {i: cat for cat, i in CATEGORY_MAP.items()}

class MinimalPrediction(BaseModel):
    id: int = Field(description="Category ID (0-7)", ge=0, le=7)

EXAMPLE_POST = "My partner won't meet my friends and I'm feeling very frustrated."
EXAMPLE_LABEL = "Love & Dating"

def classify_fast(text, title=""):
    category_list = "\n".join([f"  {i}: {cat}" for i, cat in enumerate(CATEGORIES)])
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"Classify text into one category by ID:\n{category_list}"},
                {"role": "user", "content": f"Title: {title}\nText: {text[:400]}"},
                {"role": "assistant", "content": json.dumps({"id": CATEGORY_MAP[EXAMPLE_LABEL]})},
                {"role": "user", "content": f"Text: {text[:400]}"},
            ],
            format=MinimalPrediction.model_json_schema(),
            stream=False
        )
        parsed = MinimalPrediction.model_validate_json(response['message']['content'])
        return ID_TO_CATEGORY.get(parsed.id, "Unknown")
    except Exception as e:
        print(f"    Error: {e}")
        return None

# Test on 10 posts
fs_results = []
sample_fs = df.head(10).copy()
for i, row in sample_fs.iterrows():
    pred = classify_fast(row['text_clean'], row.get('title', ''))
    if pred:
        fs_results.append({'true': row['label'], 'pred': pred})

if len(fs_results) >= 5:
    fs_df = pd.DataFrame(fs_results)
    fs_acc = accuracy_score(fs_df['true'], fs_df['pred'])
    print(f"  Few-shot accuracy: {fs_acc:.4f} ({len(fs_results)}/10)")
    if acc > 0:
        print(f"  Improvement over zero-shot: {fs_acc - acc:+.4f}")
    print("  ✓ Few-shot OK")
else:
    fs_acc = 0.0
    print(f"  ⚠ Only {len(fs_results)} successful")

# ── 7. Structured extraction ──
print("\n[7/7] Structured extraction...")
from typing import List

class ArticleAnalysis(BaseModel):
    title: str = Field(description="A concise title")
    summary: str = Field(description="2-3 sentence summary")
    institutions: List[str] = Field(description="Organizations mentioned")
    key_claims: List[str] = Field(description="Main claims (3-5)")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(description="Overall sentiment")
    topics: List[str] = Field(description="Main topics")

article = """MIT researchers have demonstrated that AI models can now perform complex reasoning
tasks previously thought to require human intelligence. The CSAIL team showed that large language
models achieve 95% accuracy on multi-step math, up from 60% two years ago. Stanford's HAI
critics argue benchmarks don't capture true understanding."""

try:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Extract structured information from the article."},
            {"role": "user", "content": article},
        ],
        format=ArticleAnalysis.model_json_schema(),
        stream=False
    )
    parsed = ArticleAnalysis.model_validate_json(response['message']['content'])
    print(f"  Title: {parsed.title}")
    print(f"  Sentiment: {parsed.sentiment}")
    print(f"  Institutions: {parsed.institutions}")
    extraction_ok = True
    print("  ✓ Extraction OK")
except Exception as e:
    print(f"  ✗ Extraction failed: {e}")
    extraction_ok = False

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"NB03b RESULT: {'PASS' if n_success >= 5 else 'PARTIAL'}")
print(f"  Model: {MODEL_NAME}")
print(f"  Zero-shot: {n_success}/15 successful, acc={acc:.4f}")
print(f"  Few-shot:  {len(fs_results)}/10 successful, acc={fs_acc:.4f}")
print(f"  Extraction: {'OK' if extraction_ok else 'FAILED'}")
print(f"  Time: {elapsed:.1f}s")
print(f"{'=' * 60}")
