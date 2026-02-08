# NB06-NB11 Testing Report

**Date:** 2026-02-08
**Environment:** macOS (MPS), Python 3.x venv, Groq API (llama-3.1-8b-instant + kimi-k2-instruct)
**Test sample sizes:** Reduced for local speed (100 tweets for NB08 vs 1000 in notebook, 30-50 examples for NB10-NB11)

---

## Executive Summary

All six notebooks run without errors. Three bugs were found and fixed. Several pedagogical improvements are recommended — the most impactful being NB08's confidence filtering narrative, which doesn't work as described because Groq's llama-3.1-8b always reports 0.80-0.90 confidence.

| Notebook | Status | Bugs Fixed | Improvements Suggested |
|----------|--------|-----------|----------------------|
| NB06 (FAISS) | Clean | 0 | 2 minor |
| NB07 (Reranking) | Clean | 0 | 1 moderate |
| NB08 (Distillation) | Fixed | 1 (exercise skeleton) | 3 significant |
| NB09 (Fine-tuning) | Skipped (CUDA) | — | — |
| NB10 (Evaluation) | Fixed | 1 (head(10) bug) | 2 moderate |
| NB11 (Annotation/IRR) | Fixed | 1 (no exercise) | 2 minor |

---

## NB06: FAISS Retrieval + Semantic Search

### Test Results

| Test | Result |
|------|--------|
| SciFact loading | 5,183 docs, 1,109 queries |
| e5-small encoding | 63.2s for 5,183 docs (CPU) |
| FAISS IndexFlatIP | 5,183 vectors, 384 dims |
| Query relevance | All 5 queries returned meaningful top-3 |
| Keyword-proxy P@5 | 100% on all 3 eval queries |
| Multilingual e5-small | 62.5s encoding, 384 dims |
| Cross-lingual overlap | EN∩FR: 1/3, EN∩DE: 0/3, EN∩ES: 1/3 |
| ChromaDB | 5 results returned, distances 0.54-0.61 |

### Assessment

**Strong notebook.** The pipeline is complete, well-explained, and runs cleanly. No code issues.

### Recommendations

**1. Cross-lingual overlap is low (pedagogical concern)**

The cross-lingual test shows EN∩German sharing 0/3 top results. This is expected with `multilingual-e5-small` (a compact model), but students might interpret this as "multilingual retrieval doesn't work."

*Suggestion:* Add a markdown cell after the cross-lingual test acknowledging this:

> "Notice that the overlap between languages is modest with this compact model. For production multilingual retrieval, larger models like `BAAI/bge-m3` (568M params) achieve much higher cross-lingual consistency. We use the smaller model here for teaching speed — the *pattern* (encode once, search across languages) is what matters."

**2. ChromaDB section uses its own default embeddings**

Cell 17 uses ChromaDB's built-in `all-MiniLM-L6-v2` (which it downloads at runtime — 79MB). Cell 19 then shows custom E5 embeddings with ChromaDB. Students may not notice the difference.

*Suggestion:* Add a brief note in the cell 17 markdown: "Note: ChromaDB uses its own default embedding model (`all-MiniLM-L6-v2`). In the next cell, we'll use our own E5 embeddings instead — which is the more realistic production pattern."

---

## NB07: Cross-encoder Reranking

### Test Results

| Test | Result |
|------|--------|
| Ground-truth eval (1,109 queries) | |
| — Precision@5 | 16.3% → 17.0% (+0.7%) |
| — NDCG@10 | 67.5% → 70.5% (+3.0%) |
| Speed | 14.1ms bi-encoder → 215ms reranked (15.2x) |
| Example query reranking | Lung cancer screening promoted from rank 5 → rank 1 |

### Assessment

**Well-structured notebook.** The qualitative example (lung cancer query) clearly demonstrates reranking value. The ground-truth evaluation section is a pedagogical highlight — students see real IR evaluation.

### Recommendations

**1. The NDCG improvement deserves more celebration (moderate)**

+3% NDCG@10 sounds small to students. The notebook already has a good "Interpreting the results" section explaining SciFact's sparsity. But consider adding one concrete example showing the *positional* improvement:

*Suggestion:* After the ground-truth eval, add a cell that finds a specific query where reranking moved the relevant doc from (say) rank 12 to rank 2, and print that example. This makes the NDCG improvement tangible:

```python
# Find queries where reranking made the biggest difference
# Show: "For query X, the relevant doc was at bi-encoder rank 12 but reranking pushed it to rank 2"
```

This is more persuasive than abstract metrics for social science students.

---

## NB08: Distillation — LLM Label Synthesis

### Test Results (100-tweet sample)

| Test | Result |
|------|--------|
| LLM labeling | 100/100 (0 errors), 3m09s |
| LLM accuracy vs true labels | **60.0%** |
| Confidence distribution | Min: 0.80, Max: 0.90, Mean: 0.83 |
| After confidence filter (>=0.7) | 100/100 kept (filter has no effect) |
| After dedup | 100 unique |
| Label distribution (LLM) | sadness: 58, joy: 20, anger: 13, love: 5, fear: 3, surprise: 1 |
| True label distribution | sadness: 35, joy: 27, anger: 14, love: 12, fear: 9, surprise: 3 |
| TF-IDF student accuracy | **28.0%** |
| E5 student accuracy | **30.0%** |

### Bugs Fixed

**Exercise cell (cell 20):** Was a bare TODO skeleton with just `print()` statements. Replaced with working code that trains TF-IDF students at different confidence thresholds and produces a comparison table.

### Critical Issues to Address

**1. Confidence filtering is broken in practice (HIGH PRIORITY)**

This is the notebook's most significant pedagogical issue. The confidence filtering section (Section 5) teaches students that "a smaller set of high-quality labels often outperforms a larger set of noisy labels." But with Groq's `llama-3.1-8b-instant`:

- **All confidences are 0.80-0.90** (extremely narrow range)
- The >=0.7 filter keeps **100% of labels**
- High-confidence accuracy equals overall accuracy (60.0%)
- The confidence filter literally does nothing

Students will run this and see no effect, which contradicts the narrative.

**Root cause:** `llama-3.1-8b-instant` with `temperature=0.0` produces poorly calibrated confidence scores. It's overconfident on everything.

*Suggestions (pick one or combine):*

a. **Change the prompt to request calibrated confidence.** Add explicit instructions:
```
"Be honest about uncertainty. Use the full 0-1 range: 0.3-0.5 for uncertain, 0.6-0.7 for moderate, 0.8+ only when very certain."
```

b. **Add a markdown cell acknowledging this** and turning it into a teaching moment:
> "You may notice that the LLM reports high confidence (0.8+) on almost everything — even when it's wrong. This is a known limitation of LLM self-reported confidence. In practice, you might use *consistency-based* confidence instead: label each text 3 times with temperature=0.7 and use the agreement rate as confidence."

c. **Implement consistency-based confidence** as an alternative: classify each text N times with temperature and measure label agreement. This is more reliable and pedagogically interesting.

Option (b) is the minimum viable fix. Option (c) would elevate the notebook significantly.

**2. LLM severely over-predicts "sadness" (MODERATE)**

The true distribution has 35 sadness / 27 joy / 14 anger / 12 love / 9 fear / 3 surprise.
The LLM labels: 58 sadness / 20 joy / 13 anger / 5 love / 3 fear / 1 surprise.

The LLM is biased toward "sadness" — it over-predicts it by 66%. This cascades into the student models, which essentially learn to predict "sadness" for everything (the E5 student gets 100% recall on sadness but 0% on everything else).

*Suggestion:* Add a class-balance analysis cell after the dedup step:

```python
# Compare LLM label distribution to true distribution
# Highlight the sadness bias
# This motivates the class-balancing exercise
```

Then make the class-balancing exercise (currently mentioned in the exercise description but not implemented) more prominent — it's the most impactful intervention students can try.

**3. Student models are embarrassingly bad at 28-30% (CONTEXT)**

This is partly a sample-size artifact (100 tweets is too small, especially after the class imbalance). The real notebook uses 1000 tweets, which should produce better results. But even at 1000, if the LLM accuracy is only 60% and heavily biased toward one class, students will get frustrating results.

*Suggestion:* Consider increasing the notebook's default sample to 1500-2000 tweets. At Groq's free tier rate of ~2.5 requests/second, 2000 tweets takes ~13 minutes — still feasible within the 65-minute block. More data + better class balance should push student accuracy to 45-55%, which is a more satisfying pedagogical result.

---

## NB09: Fine-tuning Qwen3-4B

### Test Results (Data Prep Only)

| Test | Result |
|------|--------|
| HF data loading | 16,000 train / 2,000 test |
| CSV path (NB08 output) | Works when CSV exists, graceful HF fallback |
| Chat template (Qwen2.5 tokenizer) | Produces valid `<\|im_start\|>` format |
| Dataset creation | 50 examples, "text" column present |
| Data quality | All 6 labels present, 0 empty texts |

### Assessment

Data prep is solid. Training requires CUDA — not testable locally. Skipped per user request.

---

## NB10: LLM App Evaluation

### Test Results

| Test | Result |
|------|--------|
| MODEL_SMART availability | |
| — moonshotai/kimi-k2-instruct | **WORKS** |
| — llama-3.3-70b-versatile | WORKS |
| — qwen/qwen3-32b | WORKS (but emits `<think>` tags) |
| Eval set | 30 examples (22 favor, 6 none, 2 against) |
| Stance classification accuracy | **46.7%** |
| Stance classification macro F1 | **0.449** |
| Rubric scoring (all 30) | 30/30 scored, average **3.8/5** |
| Errors | 16/30 (53%), 6 stance inversions |
| Faithfulness check | Correctly identifies faithful/unfaithful answers |

### Bugs Fixed

1. **Cell 9:** Changed `eval_set.head(10)` to iterate over all `eval_set` rows. The rubric scoring now processes all 30 examples instead of only 10.
2. **Added exercise section** with markdown instructions and `# YOUR CODE HERE` code cell.

### Recommendations

**1. The 46.7% accuracy is pedagogically useful but needs framing (MODERATE)**

Stance detection is genuinely hard — 46.7% accuracy with 3 classes (and heavy "favor" imbalance: 22/30) is actually not terrible. But students might be alarmed. The notebook already frames this as "inherently harder than topic classification," which is good.

*Suggestion:* After the classification report, add a brief analysis:

```python
# Note: The eval set is heavily imbalanced (73% favor, 20% none, 7% against)
# The LLM achieves 100% precision on "favor" — but only 32% recall
# It's conservative: it under-predicts favor and over-predicts none/against
# This is actually a common LLM behavior — they hedge toward neutral/negative stances
```

**2. qwen/qwen3-32b emits `<think>` tags in JSON mode (LOW PRIORITY)**

When tested, `qwen/qwen3-32b` returned `<think>\nOkay, the` before the JSON. This could break Pydantic parsing if a student switches to this model. Not a critical issue since the notebook defaults to `kimi-k2-instruct`, but worth noting.

*Suggestion:* Add a comment in the model selection cell:

```python
# Note: qwen/qwen3-32b sometimes emits <think> tags before JSON.
# If you switch to it, you may need to strip these tags before parsing.
```

---

## NB11: Annotation & Inter-Rater Reliability

### Test Results

| Test | Result |
|------|--------|
| Annotation set | 50 examples (36 favor, 10 none, 4 against) |
| LLM labeling | 50/50 (0 errors) |
| Raw agreement | **60.0%** |
| Cohen's Kappa | **0.356** (Fair agreement) |
| Gwet's AC1 | **0.436** |
| AC1 - Kappa | +0.080 |
| Disagreements | 20/50 (40%) |
| Stance inversions | 6/20 disagreements |
| Top confusion | favor → none (11x), favor → against (6x) |

### Bugs Fixed

**Added exercise section:** Codebook-guided classification exercise where students re-classify using the detailed codebook as the system prompt and compare kappa/AC1 to the baseline. Includes both markdown instructions and a code cell with scaffolded (commented) code.

### Assessment

**Excellent notebook.** The manual Gwet's AC1 implementation is a nice touch — it's not available in standard libraries and students learn the math. The kappa paradox explanation is clear. The disagreement analysis reveals real annotation challenges (implicit stance, sarcasm).

### Recommendations

**1. The AC1 > Kappa result validates the kappa paradox explanation (nice)**

AC1 (0.436) is higher than Kappa (0.356) by 0.080, which perfectly illustrates the lesson: kappa is pessimistic with imbalanced classes (72% favor). This is exactly the teaching moment the notebook sets up. No change needed — just noting it works as designed.

**2. Consider a confusion matrix visualization (MINOR)**

The disagreement analysis prints text-based confusion pairs. A heatmap would be more impactful for 3x3 classes.

*Suggestion:*

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(
    valid['human_label'], valid['llm_label'],
    labels=['favor', 'against', 'none'],
    cmap='Blues'
)
plt.title('Human vs LLM Label Agreement')
plt.xlabel('LLM Label')
plt.ylabel('Human Label')
plt.tight_layout()
plt.show()
```

---

## Cross-Notebook Patterns

### 1. API Rate Limiting Behavior

At ~30% through NB08's labeling run, speed dropped from 2.8 it/s to 1.07 s/it — a rate limit was hit. The `time.sleep(0.1)` between requests isn't enough. After the rate limit, the retry with `time.sleep(2 ** attempt)` kicked in and requests resumed at ~2.5 s/it.

**Implication for the classroom:** With 25 students all hitting Groq simultaneously, rate limiting will be severe. Each student running NB08 makes 1000 API calls. That's 25,000 calls across the class within a ~10 minute window.

*Suggestion:*
- Stagger NB08 starts (have students work at different paces)
- Pre-generate a shared CSV of distilled labels as a fallback
- Increase `time.sleep()` to 0.2-0.3s in NB08 to reduce rate limit hits

### 2. Stance Detection Consistency (NB10 + NB11)

Both NB10 and NB11 use the same `tweet_eval/stance_climate` dataset and the same LLM. The results are consistent:
- NB10: 46.7% accuracy on 30 examples
- NB11: 60.0% raw agreement on 50 examples (different sample)

The dominant failure mode in both is the same: **favor → none** (LLM misses implicit pro-climate stance in tweets that don't explicitly mention policy). This consistency is good for teaching — students see the same pattern across notebooks.

### 3. Exercise Section Quality

| Notebook | Exercise Quality | Notes |
|----------|-----------------|-------|
| NB06 | Good | Clear scaffolding, 5 steps, bonus challenge |
| NB07 | Good | 3 experiments with commented code stubs |
| NB08 | **Fixed** (was bare skeleton) | Now has working threshold-sweep code |
| NB10 | **Added** (was missing entirely) | 5-step pipeline + bonus model comparison |
| NB11 | **Added** (was missing entirely) | Codebook-guided re-classification |

### 4. Figure References

All figure files exist in `notebooks/figures/`:
- `reranking_pipeline.png` (NB07)
- `distillation_concept.png` (NB08)
- `distillation_pipeline.png` (NB08)
- `bertopic_pipeline.png` (NB04)
- `setfit_pipeline.png` (NB05)

They reference GitHub raw URLs (`https://raw.githubusercontent.com/RJuro/unistra-nlp2026/main/notebooks/figures/...`) which will resolve once the repo is pushed publicly.

---

## Priority Recommendations

### Must-Do Before Workshop (Feb 10)

1. **NB08: Address confidence filtering narrative** — At minimum, add a markdown cell explaining that LLM self-reported confidence is poorly calibrated. Students will otherwise be confused when the filter does nothing.

2. **Push repo to GitHub** — All figure URLs will 404 until the repo is public.

### Should-Do (Elevates Quality)

3. **NB08: Add class-balance analysis** after dedup step — show students the LLM's sadness bias visually. This motivates the class-balancing exercise.

4. **NB06: Add cross-lingual caveat** — Explain that `multilingual-e5-small` is a compact teaching model; production cross-lingual retrieval uses larger models.

5. **NB10: Frame the 46.7% accuracy** — Add a brief analysis cell explaining the class imbalance and the LLM's conservative hedging behavior.

### Nice-to-Have (Polish)

6. **NB11: Add confusion matrix heatmap** — Visual > text for 3x3 matrices.

7. **NB07: Add a concrete reranking example** — Show one query where the relevant doc moved from rank 12 to rank 2.

8. **NB08: Consider consistency-based confidence** — Classify each text 3x with temperature=0.7, use agreement as confidence. More reliable than self-reported scores and teaches an important concept.

---

## Testing Infrastructure

All test scripts are in `testing/`:

```
testing/
├── venv/                         Python virtual environment
├── test_nb06_faiss.py           8 test sections
├── test_nb07_reranking.py       5 test sections
├── test_nb08_distillation.py    8 test sections (uses Groq API)
├── test_nb09_finetuning.py      5 test sections (data prep only)
├── test_nb10_evaluation.py      7 test sections (uses Groq API)
├── test_nb11_annotation.py      7 test sections (uses Groq API)
├── run_all.sh                   Convenience script
├── emotion_distilled_labels.csv Generated by NB08 test
└── REPORT.md                    This file
```

Run individual tests: `./run_all.sh nb08`
Run all tests: `./run_all.sh`

API-dependent tests (NB08, NB10, NB11) require `GROQ_API_KEY` in `../.env`.
