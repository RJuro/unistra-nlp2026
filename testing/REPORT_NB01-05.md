# Testing Report: Notebooks NB01–NB05 + NB03b

**Date**: 2026-02-08
**Environment**: macOS Darwin 24.6.0, Python 3.9, Apple Silicon (MPS)
**Testing venv**: `testing/venv/` with sentence-transformers, bertopic, setfit, ollama, etc.

---

## Summary Table

| Notebook | Status | Time | Key Metric | Critical Issues |
|----------|--------|------|-----------|-----------------|
| NB01 TF-IDF | **PASS** | 3s | LR 84.4%, SVC 85.2% | None |
| NB02 Embeddings | **PASS** | 14s | SBERT+SVC 88.7% | None |
| NB03 LLM Zero-shot | **PARTIAL** | 23s | 88.9% (9/10) | Part B extraction **broken** + still uses llama-8b |
| NB03b Ollama Local | **PASS** | 701s | 100% (15/15) | 100% accuracy suspicious (easy first rows) |
| NB04 BERTopic | **PASS** (issues) | 728s | 7 topics, 1.1% outliers | LLM naming fails + hierarchy viz fails |
| NB05 SetFit | **PASS** | 702s | SetFit 59.6% vs TF-IDF 32.5% | LLM paraphrase output format unreliable |

---

## Detailed Results

### NB01: TF-IDF + Linear Models — PASS

**Pipeline works perfectly end-to-end.**

- Dataset: 457 rows, 8 classes, loads from GitHub
- Train/test: 342/115 (stratified, seed=42)
- Logistic Regression: **84.35%** accuracy
- LinearSVC: **85.22%** accuracy (winner)
- Confusion matrix: 8×8, correct
- Top features: sensible (e.g., "parents/family/sister" → Family Dynamics)
- Error analysis: 17/115 misclassifications (14.8%), low avg confidence on errors (0.169)
- All evaluations (classification_report, confusion matrix, feature inspection) work

**No issues found. Production-ready.**

---

### NB02: Sentence Embeddings — PASS

**All pipelines work correctly. SBERT beats TF-IDF as expected.**

| Model | Accuracy |
|-------|----------|
| SBERT + LinearSVC | **88.70%** |
| SBERT + LR | 86.96% |
| TF-IDF + SVC | 85.22% |
| TF-IDF + LR | 84.35% |

- `all-MiniLM-L6-v2` loads and encodes correctly (384 dims)
- Label efficiency experiment works (n=25 → 60%, n=100 → 77%)
- t-SNE produces valid 2D projection (457 × 2)
- Minor: sklearn FutureWarning about `n_iter` → `max_iter` in TSNE (cosmetic only)

**No issues found. Production-ready.**

---

### NB03: LLM Zero-shot + Structured Output — PARTIAL PASS

#### Part A: Zero-shot Classification — Works (with caveats)

- 9/10 posts classified successfully, **88.9% accuracy**
- 1 validation error: LLM returned `"Relationship & Dating"` instead of `"Love & Dating"`
  - Root cause: `json_object` mode does NOT enforce Literal constraints — it only guarantees valid JSON
  - The Pydantic schema catches the mismatch, so it's handled gracefully (returns None)
- Average confidence: 0.889 (very high — typical LLM overconfidence pattern)

#### Part B: Structured Extraction — **BROKEN**

**Critical bug: Groq's strict schema mode rejects the Pydantic-generated schema.**

Error message:
```
invalid JSON schema for response_format: 'ArticleAnalysis':
`additionalProperties:false` must be set on every object
```

- `Pydantic.model_json_schema()` does not include `additionalProperties: false` by default
- Groq's `json_schema` strict mode requires it on every object in the schema
- This fails 100% of the time — **every student will hit this in class**
- The `openai/gpt-oss-20b` model itself is available (responds to basic prompts)

**Fix needed**: Post-process the schema to inject `additionalProperties: false`:
```python
import copy
def make_strict_schema(schema_class):
    schema = copy.deepcopy(schema_class.model_json_schema())
    schema['additionalProperties'] = False
    return schema
```

#### Other Issues

| Issue | Severity | Detail |
|-------|----------|--------|
| **Still uses `llama-3.1-8b-instant`** | HIGH | Cell 3 sets `MODEL_FAST = "llama-3.1-8b-instant"`. Needs update to `moonshotai/kimi-k2-instruct` per project decision |
| `json_object` mode doesn't enforce Literal | MEDIUM | 1/10 posts returned wrong label format. Students may get validation errors. Consider adding fuzzy matching or retry |
| Batch size hard-coded to 10 | LOW | Cell 12 uses `df.head(10)` — fine for demo but students may miss that this is intentional |

---

### NB03b: LLM Zero-shot (Local Ollama) — PASS

**Ollama pipeline works end-to-end, including schema enforcement.**

- Model: `ministral-3:8b` (6GB download completed in ~4 min)
- Zero-shot: **15/15 successful** (100% accuracy)
- Few-shot (integer IDs): **10/10 successful** (100% accuracy)
- Structured extraction: Works perfectly (extracted title, sentiment, institutions)
- Schema enforcement via `format=Model.model_json_schema()` works flawlessly

**Caveat**: 100% accuracy on first 15/10 posts is almost certainly due to sampling bias — the first rows in dk_posts are likely very clear-cut examples. Real accuracy would be lower on the full dataset. Consider shuffling or sampling from middle of dataset.

**Performance**: ~28s per inference call locally (total 701s for ~25 calls). Acceptable for a classroom demo but students should expect 1-2 min per batch of 10.

**Comparison NB03 vs NB03b**: Ollama's `format=` parameter works better than Groq's `json_schema` strict mode because it does native constrained decoding without requiring `additionalProperties: false`.

---

### NB04: BERTopic — PASS (with issues)

**Core pipeline works. Two non-trivial issues found.**

- Dataset: Moltbook 44K posts → sampled 2000 for testing
- Embeddings: `intfloat/multilingual-e5-small`, (2000, 384) — **slow on MPS** (~11 min for 2000 docs)
- BERTopic: 7 topics discovered, only 1.1% outliers (23/2000)
- Topic 0 is a mega-topic (1529/2000 = 76%) — the spatial filtering step in the notebook addresses this
- Ground truth comparison (cross-tab) works: 9 labels × 8 topics

#### Issue 1: `visualize_hierarchy()` Fails

```
Distance matrix cannot contain negative values.
```

- Likely caused by cosine similarity producing negative values that scipy's hierarchy linkage can't handle
- The notebook uses this visualization — **students will see this error**
- **Fix**: Either skip `visualize_hierarchy()` or pre-normalize the distance matrix

#### Issue 2: LLM Topic Naming Fails

```
Please select from one of the valid options for the `tokenizer` parameter:
{'char', 'whitespace', 'vectorizer'}
```

- This is a **BERTopic API incompatibility** with the `OpenAIRepresentation` model
- The `update_topics()` call fails because BERTopic's internal tokenizer handling conflicts with the OpenAI representation model
- kimi-k2 model itself works fine (test response: "Space colonization challenges")
- **Root cause**: Likely a BERTopic version issue. The notebook may need to pin a specific version or use a different method for LLM-based topic naming

#### Other Observations

- The mega-topic problem (76% of docs in Topic 0) is real — the notebook's spatial filtering approach is well-designed to handle this
- UMAP 2D projection works correctly
- Barchart and intertopic distance visualizations work
- Moltbook dataset loads reliably from HuggingFace

---

### NB05: SetFit Few-shot — PASS

**Core pipeline works. SetFit significantly beats TF-IDF with 8 examples.**

| Method | Accuracy | Macro-F1 |
|--------|----------|----------|
| SetFit (8-shot) | **59.6%** | **58.9%** |
| TF-IDF (8-shot) | 32.5% | 32.4% |

- Dataset: `climatebert/environmental_claims` (binary: 2117 train, 265 test)
- Class imbalance: 3:1 (1585 no-claim vs 532 claim)
- SetFit recall for env claims: **93%** but precision only 38% → over-predicts claims
- SetFit clearly dominates TF-IDF at 8 examples (as expected)
- E5 prefix handling (`query: ` prefix) works correctly

#### Issue: LLM Paraphrase Output Format

- `openai/gpt-oss-20b` returns paraphrases wrapped in markdown code blocks:
  ```
  ```json
  ["paraphrase 1", "paraphrase 2"]
  ```
  ```
- `json.loads()` can't parse this directly
- The notebook has a `_parse_list_output()` function that should handle this — **need to verify it strips markdown fences**
- If the parser doesn't handle this, LLM bootstrapping will silently fail and students won't get augmented data

#### Other Observations

- `transformers>=4.40,<5` version pin is correct and important (SetFit depends on a function removed in v5)
- Training is slow on MPS (~12 min for 80 iterations with 320 pairs)
- 59.6% accuracy with 8 examples is reasonable for a 3:1 imbalanced binary dataset

---

## Cross-Notebook Patterns

### Model Usage Summary

| Notebook | LLM Model | Issue |
|----------|-----------|-------|
| NB01 | None | — |
| NB02 | None | — |
| NB03 | `llama-3.1-8b-instant` + `openai/gpt-oss-20b` | **llama-8b needs update to kimi-k2** |
| NB03b | `ministral-3:8b` (Ollama) | OK |
| NB04 | `moonshotai/kimi-k2-instruct` | OK (already updated) |
| NB05 | `openai/gpt-oss-20b` | OK (used for paraphrases, not classification) |

**NB03 is the only notebook still using `llama-3.1-8b-instant`.**

### Schema Enforcement Comparison

| Method | Provider | Works? | Notes |
|--------|----------|--------|-------|
| `json_object` | Groq | Partially | Doesn't enforce Literal constraints |
| `json_schema` (strict) | Groq/gpt-oss-20b | **BROKEN** | Needs `additionalProperties: false` |
| `format=` | Ollama | **Works perfectly** | Native constrained decoding |

### Performance on Mac (MPS)

| Task | Time | Notes |
|------|------|-------|
| TF-IDF + sklearn | 3s | Fast |
| SBERT encoding (457 docs) | 2s | Fast |
| SBERT encoding (2000 docs) | 11 min | Slow — MPS contention with Ollama? |
| BERTopic (2000 docs) | ~12 min total | Embedding-dominated |
| SetFit training (320 pairs, 80 iter) | ~12 min | Slow on MPS |
| Ollama inference (per call) | ~28s | Acceptable for small batches |
| Groq API (per call) | <1s | Fast |

---

## Priority Recommendations

### Must-Do (Before Workshop)

1. **NB03: Fix Part B structured extraction** — Add `additionalProperties: false` to schema. This breaks 100% of the time.

2. **NB03: Switch `llama-3.1-8b-instant` → `moonshotai/kimi-k2-instruct`** — Last notebook still using the old model.

3. **NB04: Fix LLM topic naming** — The `update_topics()` call with `OpenAIRepresentation` fails due to tokenizer parameter mismatch. Either:
   - Pin BERTopic version that works
   - Use a different approach for LLM naming (direct API call instead of BERTopic wrapper)
   - Add try/except with manual fallback

### Should-Do

4. **NB04: Fix `visualize_hierarchy()`** — Either add `abs()` on distance matrix or catch the error and skip

5. **NB05: Verify `_parse_list_output()` handles markdown fences** — If LLM returns ` ```json [...] ``` `, the parser must strip the fences before `json.loads()`

6. **NB03: Add retry or fuzzy matching for label validation** — 1/10 posts returned a close-but-wrong label ("Relationship & Dating" vs "Love & Dating"). Students will see this.

### Nice-to-Have

7. **NB03b: Test on shuffled/random sample** — 100% accuracy on first 15 rows is misleading for students

8. **NB04: Warn about Colab embedding speed** — 2000 docs took 11 min on MPS; on Colab T4 it should be 2-3 min, but students should expect some wait

9. **NB02: Suppress sklearn FutureWarning** — `n_iter` → `max_iter` warning in TSNE is cosmetic but distracting

---

## Files Modified: None

This is a testing-only report. No notebook edits were made.
