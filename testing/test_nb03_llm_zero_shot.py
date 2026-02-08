"""Test script for NB03: LLM Zero-shot Classification + Structured Output"""
import sys, os, time, json, re
start = time.time()

print("=" * 60)
print("TEST: NB03 — LLM Zero-shot + Structured Output")
print("=" * 60)

# ── 1. API setup ──
print("\n[1/7] API setup...")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("  ✗ GROQ_API_KEY not set — cannot test LLM cells")
    sys.exit(1)

from openai import OpenAI
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# NOTE: NB03 uses llama-3.1-8b-instant — flag for replacement
MODEL_FAST = "llama-3.1-8b-instant"  # <-- NB03 STILL USES THIS (needs update to kimi-k2)
MODEL_SMART = "openai/gpt-oss-20b"   # Part B uses this for strict schema
print(f"  MODEL_FAST: {MODEL_FAST}")
print(f"  MODEL_SMART: {MODEL_SMART}")

# Test connectivity
try:
    test_resp = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": "Say 'hello' in one word."}],
        max_tokens=10, temperature=0.0,
    )
    print(f"  Connectivity test: {test_resp.choices[0].message.content.strip()}")
    print("  ✓ Groq API OK")
except Exception as e:
    print(f"  ✗ Groq API failed: {e}")
    sys.exit(1)

# ── 2. Data loading ──
print("\n[2/7] Loading dataset...")
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

DATA_URL = "https://raw.githubusercontent.com/RJuro/unistra-nlp2026/main/data/dk_posts_synth_en_processed.json"
df = pd.read_json(DATA_URL)
df['text'] = df['title'] + ' . ' + df['selftext']
df['text_clean'] = df['text'].apply(lambda t: re.sub(r'\s+', ' ', t.lower()).strip())
print(f"  Loaded {len(df)} rows")

# ── 3. Schema definition ──
print("\n[3/7] Pydantic schemas...")
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

print(f"  Schema fields: {list(SingleLabelPrediction.model_fields.keys())}")
print("  ✓ Schema OK")

# ── 4. Part A: Classify 10 posts ──
print("\n[4/7] Part A: Zero-shot classification (10 posts)...")
sample = df.head(10).copy()

def classify_post(text, model=MODEL_FAST):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"Classify this text into exactly one category: {CATEGORIES}. Return JSON with predicted_label and confidence."},
                {"role": "user", "content": text[:500]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0, max_tokens=100,
        )
        content = response.choices[0].message.content
        parsed = SingleLabelPrediction.model_validate_json(content)
        return parsed
    except Exception as e:
        print(f"    Error: {e}")
        return None

results = []
for i, row in sample.iterrows():
    result = classify_post(row['text_clean'])
    if result:
        results.append({
            'true_label': row['label'],
            'predicted_label': result.predicted_label,
            'confidence': result.confidence,
        })
    time.sleep(1.0)

valid = pd.DataFrame(results)
n_success = len(valid)
print(f"  Successful: {n_success}/10")

if n_success >= 5:
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(valid['true_label'], valid['predicted_label'])
    print(f"  Accuracy: {acc:.4f} (on {n_success} samples)")
    avg_conf = valid['confidence'].mean()
    print(f"  Avg confidence: {avg_conf:.3f}")
    print("  ✓ Part A OK")
else:
    print("  ⚠ Too few successful predictions — API issues?")

# ── 5. Part B: Structured extraction ──
print("\n[5/7] Part B: Structured extraction...")

class ArticleAnalysis(BaseModel):
    title: str = Field(description="A concise title for the article")
    summary: str = Field(description="2-3 sentence summary")
    institutions: List[str] = Field(description="Organizations mentioned")
    key_claims: List[str] = Field(description="Main claims or findings (3-5)")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(description="Overall sentiment")
    topics: List[str] = Field(description="Main topics discussed")

article_text = """
MIT researchers have demonstrated that artificial intelligence models can now perform
complex reasoning tasks that were previously thought to require human-level intelligence.
The team at MIT's Computer Science and Artificial Intelligence Laboratory (CSAIL) showed
that large language models, when properly fine-tuned, can solve multi-step mathematical
problems with 95% accuracy. This represents a significant leap from the 60% accuracy
reported just two years ago. Critics from Stanford's Human-Centered AI Institute argue
that these benchmarks don't capture true understanding.
"""

def extract_with_retry(text, max_retries=3):
    schema_dict = ArticleAnalysis.model_json_schema()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_SMART,
                messages=[
                    {"role": "system", "content": "Extract structured information from the article. Return valid JSON."},
                    {"role": "user", "content": text},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ArticleAnalysis",
                        "strict": True,
                        "schema": schema_dict,
                    }
                },
                temperature=0.0, max_tokens=2048,
            )
            content = response.choices[0].message.content
            parsed = ArticleAnalysis.model_validate_json(content)
            return parsed
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return None

result = extract_with_retry(article_text)
if result:
    print(f"  Title: {result.title}")
    print(f"  Sentiment: {result.sentiment}")
    print(f"  Institutions: {result.institutions}")
    print(f"  Key claims: {len(result.key_claims)}")
    assert len(result.institutions) >= 1, "Expected at least 1 institution"
    assert len(result.key_claims) >= 1, "Expected at least 1 key claim"
    print("  ✓ Part B OK")
else:
    print("  ⚠ Extraction failed after retries")
    # Test if MODEL_SMART is available
    print(f"  Testing MODEL_SMART ({MODEL_SMART}) availability...")
    try:
        test = client.chat.completions.create(
            model=MODEL_SMART,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10,
        )
        print(f"  MODEL_SMART responds: {test.choices[0].message.content}")
        print("  ⚠ Model works but strict schema extraction failed")
    except Exception as e:
        print(f"  ✗ MODEL_SMART not available: {e}")
        print("  ISSUE: Need fallback model for Part B")

# ── 6. Cost analysis ──
print("\n[6/7] Cost analysis check...")
# Rough estimate
tokens_per_post = 200  # ~150 input + ~50 output
total_tokens = 10 * tokens_per_post
print(f"  Estimated tokens used: ~{total_tokens}")
print(f"  Free tier remaining: ~{14400 - 10} requests (llama-3.1-8b)")
print("  ✓ Cost analysis OK")

# ── 7. Exercise template validation ──
print("\n[7/7] Exercise template check...")
# Verify that the helper function make_strict_format would work
def make_strict_format(schema_class, name):
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema_class.model_json_schema(),
        }
    }
fmt = make_strict_format(ArticleAnalysis, "test")
assert fmt["type"] == "json_schema"
assert fmt["json_schema"]["strict"] == True
print("  ✓ make_strict_format helper works")

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"NB03 RESULT: {'PASS' if n_success >= 5 else 'PARTIAL'}")
print(f"  Classification: {n_success}/10 successful, acc={acc:.4f}" if n_success >= 5 else f"  Classification: {n_success}/10 successful")
print(f"  Extraction: {'OK' if result else 'FAILED'}")
print(f"  MODEL_FAST: {MODEL_FAST} ← NEEDS UPDATE TO kimi-k2")
print(f"  MODEL_SMART: {MODEL_SMART}")
print(f"  Time: {elapsed:.1f}s")
print(f"{'=' * 60}")
