# UNISTRA NLP 2026 Workshop — Full Specification

## 1. Schedule

| Block | Day | Time | Hours | Theme |
|-------|-----|------|-------|-------|
| 1 | Tue Feb 10 AM | 08:00–12:00 | 4h | Kickoff Presentation + TF-IDF Baselines + Sentence Embeddings |
| 2 | Tue Feb 10 PM | 13:00–17:00 | 4h | LLM Zero-shot + BERTopic + Sprint 1 |
| 3 | **Wed Feb 11 AM** | **08:00–12:00** | 4h | SetFit Few-shot + FAISS Retrieval + Eval Habits |
| 4 | Thu Feb 12 AM | 08:00–12:00 | 4h | Reranking + Distillation + Fine-tuning |
| 5 | Thu Feb 12 PM | 13:00–17:00 | 4h | Annotation/IRR + Final Sprint + Presentations |

**Total: 20h** — Room A 330, Idem-Lab, University of Strasbourg
**Note**: Students realistically arrive ~08:20. All morning blocks budget for a soft 08:20 start.

---

## 2. Session-by-Session Plan

### Block 1 — Tuesday Morning (08:20–12:00): Foundation

| Time | Activity | Duration |
|------|----------|----------|
| 08:20–08:50 | **Kickoff presentation** (LaTeX Beamer): NLP landscape, course structure, tools, project menu. See §3A below. | 30 min |
| 08:50–10:20 | **NB01**: TF-IDF + Linear Models — "Baselines that Win". Includes evaluation (confusion matrix, classification report, error analysis). | 90 min |
| 10:20–10:35 | Break | 15 min |
| 10:35–12:00 | **NB02**: Sentence Embeddings — "Universal Features". SBERT vs TF-IDF comparison, label efficiency curves. | 85 min |

### Block 2 — Tuesday Afternoon (13:00–17:00): LLM Classification + Topics + Sprint 1

| Time | Activity | Duration |
|------|----------|----------|
| 13:00–13:50 | **NB03**: LLM-based Zero-shot Classification — using Groq API (llama-3.1-8b or qwen3-32b), structured output with Pydantic. Not DeBERTa. | 50 min |
| 13:50–14:05 | Break | 15 min |
| 14:05–15:35 | **NB04**: BERTopic — Topic discovery + LLM topic annotation via Groq. | 90 min |
| 15:35–15:50 | Break + Sprint 1 intro: "Pick dataset, pick task, define metric." Keep options tight. | 15 min |
| 15:50–16:45 | **Project Sprint 1** | 55 min |
| 16:45–17:00 | Lightning shares (2–3 groups) + day wrap | 15 min |

**Sprint 1 Deliverable**: Notebook with dataset loaded + train/val split + TF-IDF or embedding baseline + 1 metric + 5 example errors

### Block 3 — Wednesday Morning (08:20–12:00): Few-shot + Retrieval

| Time | Activity | Duration |
|------|----------|----------|
| 08:20–09:30 | **NB05**: SetFit Few-shot Classification — green trademark / sustainability claims. LLM-bootstrapped contrastive pairs. | 70 min |
| 09:30–09:45 | Break | 15 min |
| 09:45–10:55 | **NB06**: FAISS Retrieval + Semantic Search — multilingual option with bge-m3. | 70 min |
| 10:55–11:10 | Break | 15 min |
| 11:10–11:45 | **Mini-session**: Evaluation habits — macro F1, calibration, slice analysis, data leakage detection. | 35 min |
| 11:45–12:00 | Sprint 2 planning: "Add a second approach + evaluation discipline." Students plan their Thursday work. | 15 min |

### Block 4 — Thursday Morning (08:20–12:00): Advanced Pipelines

| Time | Activity | Duration |
|------|----------|----------|
| 08:20–09:10 | **NB07**: Cross-encoder Reranking — contextualized for social science retrieval (policy docs, case law, academic search). | 50 min |
| 09:10–09:25 | Break | 15 min |
| 09:25–10:30 | **NB08**: Distillation — LLM-generated training data via Groq/Together. Structured output + confidence filtering + dedup. | 65 min |
| 10:30–10:45 | Break | 15 min |
| 10:45–12:00 | **NB09**: Fine-tuning a Small LM — LoRA/QLoRA via Unsloth with **Qwen3-4B** (fits Colab T4). | 75 min |

### Block 5 — Thursday Afternoon (13:00–17:00): Evaluation + Annotation + Final Sprint

| Time | Activity | Duration |
|------|----------|----------|
| 13:00–13:40 | **NB10**: LLM App Evaluation — rubric-based scoring, automated metrics, qualitative analysis. | 40 min |
| 13:40–14:10 | **NB11**: Annotation & IRR — Argilla intro, human vs LLM labeling, inter-rater reliability (Cohen's κ / Gwet's AC1), deductive coding workflow. | 30 min |
| 14:10–14:20 | Sprint intro: "Finish a coherent pipeline + show evidence" | 10 min |
| 14:20–15:50 | **Project Sprint 2** (final) | 90 min |
| 15:50–16:00 | Break + prep | 10 min |
| 16:00–16:50 | **Final share-out**: 5-min demos per group | 50 min |
| 16:50–17:00 | Course wrap-up + feedback | 10 min |

**Sprint 2 Deliverable**: Best model/pipeline + clean evaluation + 3–5 qualitative examples + 1 short "model card" (data, method, limits, failure modes)

---

## 3. Deliverables to Create

### 3A. Kickoff Presentation (LaTeX Beamer)

**Goal**: 15–20 slides introducing the NLP landscape and the course.

**Style**: Match the webpage color scheme — dark warm background (#2C2421), cream text (#FAF7F2), coral highlights (#E07850), gold accents (#D4A855). Clean academic style with cited images.

**Content outline**:
1. Title slide: "Applied NLP: From Text to Intelligence" / University of Strasbourg / Feb 2026
2. Instructor intro
3. The NLP task landscape (classification, extraction, similarity, generation, topic modeling) — with diagram
4. The 5 eras of text representation (from M2_NLP_intro.ipynb): regex → TF-IDF → Word2Vec → BERT → LLMs — with timeline figure
5. Where we are now: foundation models, embeddings everywhere, few-shot learning
6. Course roadmap: visual of the 5 blocks
7. Tools & setup: Groq, Together, Ollama, Colab
8. Project menu overview (4 tracks)
9. Datasets we'll use
10. "Let's start" transition slide

**Figures needed** (create `presentation/figures/` folder):
- NLP task taxonomy diagram (can generate or find CC-licensed)
- Timeline of text representation eras
- Embedding space visualization
- LLM landscape/architecture overview
- All images must include source citations

**Reference for style**: Check 2024 repo (`https://github.com/RJuro/unistra-nlp2024`) and existing `lecture/NLP-Conceptual-Intro.pptx` (46 MB) for Roman's preferred presentation approach.

**Files to create**:
- `presentation/kickoff.tex` (Beamer source)
- `presentation/beamerthemeUNISTRA.sty` (custom theme matching webpage colors)
- `presentation/figures/` (image assets with citations)
- `presentation/kickoff.pdf` (compiled output)

### 3B. Notebooks (11 total)

| # | Title | Duration | Source | Dataset |
|---|-------|----------|--------|---------|
| 01 | TF-IDF + Linear Models | 90 min | Adapt `old_notebooks/NLP_intro 2/M2_NLP_Supervised.ipynb` | dk_posts (Danish Reddit advice, 457 posts, 8 classes) |
| 02 | Sentence Embeddings | 85 min | Adapt `old_notebooks/NLP_intro 2/M2_NLP_Supervised.ipynb` | dk_posts (same dataset, compare approaches) |
| 03 | LLM Zero-shot + Structured Output | 50 min | Adapt `old_notebooks/Session2_notebooks_data 2/m2_nlp_llmstructure.ipynb` (Pydantic schemas, extraction pipeline) + `m2_nlp_ollama.ipynb` (zero-shot classification pattern) | dk_posts (classify) + real-world articles (extract) |
| 04 | BERTopic + LLM Topic Annotation | 90 min | Adapt `old_notebooks/NLP_intro 2/M2_NLP_TopicModelBert.ipynb` | Moltbook posts (44K AI agent posts) or podcast transcripts |
| 05 | SetFit Few-shot | 70 min | New — reference 2024's `UNISTRA-02-FewShot-classifier-argilla-setfit.ipynb` | Green claims / EUIPO trademark terms + LLM-bootstrapped pairs |
| 06 | FAISS Retrieval + Semantic Search | 70 min | New | Multilingual corpus (bge-m3), policy docs or academic abstracts |
| 07 | Cross-encoder Reranking | 50 min | New (builds on NB06) | Same corpus as NB06, social science retrieval framing |
| 08 | Distillation — LLM Label Synthesis | 65 min | Adapt batch/retry patterns from `old_notebooks/Session2_notebooks_data 2/m2_nlp_llmstructure.ipynb` + 2024's `UNISTRA-07-LLMLabelSynthesis.ipynb` | Unlabeled text → Groq labels → filtered training set |
| 09 | Fine-tuning (Unsloth + Qwen3) | 75 min | New — Unsloth Colab templates | Distilled dataset from NB08; Qwen3-4B via LoRA |
| 10 | LLM App Evaluation | 40 min | New | Eval set of ~30 examples |
| 11 | Annotation & IRR | 30 min | New — reference 2024's Argilla notebooks | Human vs LLM labels, Cohen's κ, deductive coding |

**Plus**: Structured extraction demo somewhere — podcast transcripts → structured JSON (could be woven into NB03 or NB08 as a motivating example: "why do we need structured output?")

### Notebook Design Principles

Each notebook follows a consistent structure:
1. **Header cell**: Title, learning goals, estimated time, Colab badge
2. **Setup cell**: `pip install` + imports + API key setup (Groq primary / Together fallback / Ollama local)
3. **Conceptual intro**: 2–3 markdown cells explaining the approach
4. **Guided code + exercises**: Code cells with `# YOUR CODE HERE` blocks interleaved with solutions
5. **Evaluation section**: Always compare to at least one prior approach
6. **Takeaway cell**: Summary + "when to use this in practice" guidance

### LLM Provider Strategy

```python
# Pattern used in ALL notebooks that need LLM inference
import os
from openai import OpenAI

# === Choose your provider (uncomment one) ===

# Groq (primary — free, fast)
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL = "llama-3.1-8b-instant"  # 14.4K requests/day free
# MODEL = "qwen/qwen3-32b"     # higher quality, 1K requests/day free

# Together.AI (fallback — $5 free credit)
# client = OpenAI(
#     api_key=os.environ.get("TOGETHER_API_KEY"),
#     base_url="https://api.together.xyz/v1"
# )
# MODEL = "meta-llama/Llama-3.1-8B-Instruct"
```

**Groq free tier models** (Feb 2026):
- `llama-3.1-8b-instant`: 14.4K RPD, 500K TPD — **best for classroom** (high limits)
- `qwen/qwen3-32b`: 1K RPD, 500K TPD — higher quality, for teacher demos
- `llama-3.3-70b-versatile`: 1K RPD, 100K TPD — highest quality, limited

**Ollama in Colab** (for NB08–09):
```bash
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve &
!ollama pull qwen3:4b
```

### Notebook-by-Notebook Specs

**NB01 — TF-IDF + Linear Models** (90 min)
- Adapt directly from `M2_NLP_Supervised.ipynb` TF-IDF section
- Dataset: `dk_posts_synth_en_processed.json` (457 posts, 8 categories: Love & Dating, Family Dynamics, Work/Study/Career, Friendship, Health, Finance, Practical Questions, Everyday Observations)
- Pipeline: text cleaning → TfidfVectorizer → LogisticRegression + LinearSVC
- Evaluation: classification_report, confusion matrix, per-class F1, top features inspection, 5 worst misclassifications
- Keep Roman's existing evaluation approach from the autumn notebook

**NB02 — Sentence Embeddings** (85 min)
- Adapt from `M2_NLP_Supervised.ipynb` SBERT section
- Models: `all-MiniLM-L6-v2` (default), mention `BAAI/bge-m3` for multilingual
- Pipeline: encode → LogisticRegression on embeddings vs TF-IDF
- Exercises: label efficiency curve (10, 50, 100, 500 labels), side-by-side comparison
- Key result to reproduce: SBERT+LR ≈ 87.8% vs TF-IDF+LR ≈ 84.4%

**NB03 — LLM Zero-shot + Structured Output** (50 min)
- **LLM-based, NOT DeBERTa** — use Groq API with structured output (Pydantic)
- **Adapt heavily from Session 2 notebooks**:
  - From `m2_nlp_ollama.ipynb`: zero-shot classification of dk_posts using `SingleLabelPrediction` Pydantic schema with Literal types, accuracy evaluation with sklearn, few-shot improvement pattern
  - From `m2_nlp_llmstructure.ipynb`: schema-enforced JSON extraction, the 3 real-world articles (AI scaling, VC funding, Danish public sector AI), `extract_with_retry()` pattern, cost analysis
- **Part A — Classification** (25 min): Apply LLM to dk_posts → compare accuracy to NB01/NB02 without training. Swap Ollama/Gemini for Groq API (same OpenAI-compatible pattern).
- **Part B — Structured Extraction** (25 min): Extract structured fields from a news article using Pydantic schema → show how this enables downstream analysis (DataFrame, filtering, aggregation). Motivates "why structured output matters."
- Exercise: try different prompts, vary schema complexity, measure accuracy + cost tradeoffs

**NB04 — BERTopic + LLM Topic Annotation** (90 min)
- Adapt from `M2_NLP_TopicModelBert.ipynb`, update for BERTopic v0.17+
- Dataset: **Moltbook** (44K AI agent posts, 9 content categories) — fun and topical
- Pipeline: sentence-transformers → UMAP → HDBSCAN → BERTopic
- LLM topic naming via Groq (replace Gemini from old notebook)
- Exercises: explore topics, visualize topic map, interpret clusters, compare to ground-truth categories
- Alternative dataset: podcast transcripts (Lex Fridman or SPoRC subset) for temporal analysis

**NB05 — SetFit Few-shot** (70 min)
- Reference 2024's `UNISTRA-02-FewShot-classifier-argilla-setfit.ipynb` for structure
- Dataset: `climatebert/environmental_claims` (binary: environmental claim yes/no) — green/sustainability theme
- **LLM-bootstrapped contrastive pairs**: Use Groq to generate positive/negative example pairs → feed to SetFit
- Pipeline: fine-tune `paraphrase-mpnet-base-v2` on 8–32 examples/class → logistic head
- Exercises: vary number of examples (4, 8, 16, 32), compare to NB01–03 results

**NB06 — FAISS Retrieval + Semantic Search** (70 min)
- Multilingual with `BAAI/bge-m3` (568M params, MIT, top MTEB scores)
- Dataset: policy documents, EU legislative texts, or academic abstracts — meaningful for social science students
- Pipeline: encode corpus → FAISS index → query → top-k results
- Exercises: define 10 queries, judge relevance manually, compute precision@k
- Show bilingual retrieval (query in French, retrieve English docs or vice versa)

**NB07 — Cross-encoder Reranking** (50 min)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Contextualized for social science**: frame as "finding relevant policy documents", "academic paper retrieval", "case law search"
- Pipeline: bi-encoder retrieval (NB06) → cross-encoder rerank top-k
- Exercises: measure precision@5 before/after reranking, qualitative comparison

**NB08 — Distillation / LLM Label Synthesis** (65 min)
- Reference 2024's `UNISTRA-07-LLMLabelSynthesis.ipynb` for structure
- **Reuse from Session 2 notebooks**:
  - `m2_nlp_llmstructure.ipynb`: `extract_with_retry()` with exponential backoff, batch processing with tqdm, Pydantic schema validation, cost analysis methodology
  - `m2_nlp_ollama.ipynb`: `classify_fast()` batch function, accuracy evaluation with classification_report, multilingual generalization pattern
- Pipeline: Groq/Together API (large model) labels unlabeled text → confidence filtering → dedup → class balancing → train small sklearn classifier
- Exercises: create synthetic labeled set (~500–1000 examples), compare to human-labeled baseline from NB01
- The batch processing + retry patterns from Session 2 are production-ready and can be lifted almost directly

**NB09 — Fine-tuning with Unsloth + Qwen3** (75 min)
- Library: `unsloth` + `trl`
- Model: **Qwen3-4B** via LoRA (fits free Colab T4, 2× faster with Unsloth)
- **Reuse from Session 2**: `m2_nlp_ollama.ipynb` Ollama server management pattern (`start_ollama()`, model pull, health check) for post-training local inference
- Pipeline: format distilled data as instruction pairs → LoRA fine-tune → evaluate vs zero-shot (NB03)
- Export to GGUF/Ollama format as bonus — use the Ollama patterns from Session 2 to serve the fine-tuned model locally
- New notebook (follow Unsloth Qwen3 Colab templates)

**NB10 — LLM App Evaluation** (40 min)
- Rubric-based scoring (define criteria, score 1–5)
- Automated metrics: accuracy, macro F1, faithfulness (if retrieval)
- Brief RAGAS concepts for retrieval evaluation
- Exercises: build eval set of ~30 examples, score with rubric + automated metrics

**NB11 — Annotation & IRR** (30 min)
- Argilla quick setup (HF Spaces deployment or local)
- Workflow: import misclassified examples → human label → export → compare
- Inter-rater reliability: Cohen's κ, Gwet's AC1 (as used in 2024's LLM comparison notebook)
- Human vs LLM labeling comparison: same examples labeled by both → measure agreement
- Deductive coding workflow: predefined codebook → LLM applies codes → human validates
- If Argilla doesn't install: fallback to CSV-based labeling (same concepts)

---

## 4. Datasets

### Demo Dataset (NB01–03)

**Danish Reddit Advice Posts** (`dk_posts_synth_en_processed.json`)
- 457 posts, 8 categories, English text
- Already used in old_notebooks — students get familiar with it across NB01–03
- Source: `old_notebooks/NLP_intro 2/dk_posts_synth_en_processed.json`

### Creative Project Datasets

| Dataset | Source | Size | Good For | Fun Factor |
|---------|--------|------|----------|------------|
| **Moltbook** (AI agent social network) | `TrustAIRLab/Moltbook` on HF | 44K posts, 9 categories + toxicity | Classification, topic modeling, content moderation | AI agents posting on social media — leaked database from the "front page of the agent internet" |
| **Lex Fridman Podcast** | `Whispering-GPT/lex-fridman-podcast` on HF | Full archive, Whisper transcripts | Topic modeling, temporal drift, NER | Recognizable guests, wide-ranging topics |
| **All-In Podcast** | `dfurman/All-In-Podcast-Transcripts` on HF | Full archive | Topic modeling, VC/startup culture | Tech/finance angle |
| **SEC Financial Filings** | `PleIAs/SEC` or EDGAR-CRAWLER | 1000s of 10-K filings | **Structured extraction** → econometrics via LLM | Real financial data, LLM pulls out numbers/metrics |
| **Environmental Claims** | `climatebert/environmental_claims` on HF | Binary classification | SetFit (NB05), greenwashing detection | Timely ESG topic |
| **Bluesky Posts** | `alpindale/two-million-bluesky-posts` on HF | 2M posts | Social media analysis, platform migration | Twitter→Bluesky migration study |

### Extraction Example

For "getting data out for structured analysis" — two use cases to weave into notebooks:
1. **Podcast → structured topics**: Transcript chunk → LLM extracts {topic, speakers, key_claims, sentiment} → feed to BERTopic or analysis (NB04)
2. **Financial filings → econometrics**: SEC 10-K text → LLM extracts {revenue, risk_factors, esg_mentions, forward_guidance} → structured DataFrame (NB03 or NB08)

---

## 5. Project Menu

Students pick 1 track, work across Sprint 1 (Tue PM) and Sprint 2 (Thu PM):

| Track | Description | Key Notebooks | Suggested Dataset |
|-------|-------------|---------------|-------------------|
| **A. Classification** | Classify text into categories — stance, sentiment, topic tags, toxicity | NB01–03, NB05, NB08–09 | Moltbook (toxicity), environmental claims, dk_posts |
| **B. Topic Discovery** | BERTopic + LLM annotation + interpretability | NB04, NB01–02 for comparison | Moltbook, podcast transcripts, Bluesky posts |
| **C. Semantic Search** | Build a retrieval system with reranking | NB06–07, NB02 for embeddings | Policy docs, academic papers, SEC filings |
| **D. Structured Extraction** | LLM → structured JSON → analysis pipeline | NB03, NB08 | Podcast transcripts, SEC filings, news articles |

Students can also bring their own data (thesis-related encouraged).

---

## 6. Webpage Specification

### Design

Adopt the **Governance_presentations** design language:
- **Dark warm theme**: bg `#2C2421`, cream `#FAF7F2`, coral `#E07850`, gold `#D4A855`
- **Typography**: Playfair Display (headings) + Source Sans 3 (body) + Caveat (accents)
- **Layout**: Sticky nav, hero with floating stat cards, grid sections, slide-in session panels
- **Pattern**: Subtle dot grid overlay via radial-gradient
- **Interactions**: Smooth scroll, panel slide-in (cubic-bezier), card hover lift, dark/light toggle (dark default)

### Improvements Over 2025

The 2025 site was basic shadcn-style cards. The 2026 site:
- Rich editorial design (Governance_presentations style)
- Visual timeline showing the Tue/Wed/Thu schedule
- Session panels with all resources (notebooks, Colab links, data, slides)
- Step-by-step API key setup with screenshots
- Project menu cards with dataset recommendations

### Site Structure

```
index.html (single page + panel overlays)
├── Header (sticky): "NLP/LLM 2026" logo + nav (Overview, Schedule, Sessions, Requirements)
├── Hero: Title, subtitle, instructor, "Feb 10–12, 2026 • Strasbourg"
│   └── Floating cards: "20 hours", "11 notebooks", "4 project tracks"
├── Overview: Course description + learning outcomes
├── Schedule: Visual timeline (Tue 8h → Wed 4h → Thu 8h)
├── Sessions Grid: 5 blocks (click → slide-in panel)
├── Requirements:
│   ├── Google Colab (free)
│   ├── Groq API key (free, step-by-step)
│   ├── Together.AI API key ($5 free credit)
│   ├── Ollama (optional local install)
│   └── Git clone command
├── Project Menu: 4 track cards
└── Footer: Roman Jurowetzki • Aalborg University / University of Strasbourg • 2026

Session panels (fetched on click):
├── block1.html — Tue AM content (Kickoff + NB01 + NB02)
├── block2.html — Tue PM content (NB03 + NB04 + Sprint 1)
├── block3.html — Wed AM content (NB05 + NB06 + Eval)
├── block4.html — Thu AM content (NB07 + NB08 + NB09)
└── block5.html — Thu PM content (NB10 + NB11 + Sprint 2 + Presentations)
```

Each panel: session title, time, learning goals, notebook cards with "Open in Colab" (coral button) + "Download" (ghost button) + data download link.

### Deployment

- GitHub Pages from `https://github.com/RJuro/unistra-nlp2026`
- Colab links: `https://colab.research.google.com/github/RJuro/unistra-nlp2026/blob/main/notebooks/NB01_tfidf_baselines.ipynb`

---

## 7. Repository Structure

```
unistra-nlp2026/
├── index.html
├── block1.html … block5.html
├── notebooks/
│   ├── NB01_tfidf_baselines.ipynb
│   ├── NB02_sentence_embeddings.ipynb
│   ├── NB03_llm_zero_shot.ipynb
│   ├── NB04_bertopic.ipynb
│   ├── NB05_setfit_fewshot.ipynb
│   ├── NB06_faiss_retrieval.ipynb
│   ├── NB07_reranking.ipynb
│   ├── NB08_distillation.ipynb
│   ├── NB09_finetuning_qwen3.ipynb
│   ├── NB10_llm_evaluation.ipynb
│   └── NB11_annotation_irr.ipynb
├── data/
│   ├── dk_posts_synth_en_processed.json   (demo dataset, from old_notebooks)
│   └── README.md                           (dataset descriptions + HF download instructions)
├── presentation/
│   ├── kickoff.tex
│   ├── kickoff.pdf
│   ├── beamerthemeUNISTRA.sty
│   └── figures/                            (images with source citations)
├── utils/
│   ├── eval_helpers.py
│   └── data_loader.py
├── research/                               (planning docs, not linked from site)
├── old_notebooks/                          (archive, not linked)
├── UNISTRA25/                              (archive, not linked)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 8. Technical Requirements

### API Keys

| Provider | Purpose | Free Tier | Recommended Model |
|----------|---------|-----------|-------------------|
| **Groq** | Primary LLM inference | 14.4K req/day (8b), 1K req/day (32b/70b) | `llama-3.1-8b-instant` for class, `qwen/qwen3-32b` for demos |
| **Together.AI** | Backup + more models | $5 free credit | `meta-llama/Llama-3.1-8B-Instruct` |
| **Ollama** | Local inference | Unlimited | `qwen3:4b` |

### Key Python Packages

```
openai>=1.0          # LLM API (all notebooks)
pydantic>=2.0        # Structured output
pandas, numpy, scikit-learn, matplotlib  # Core
sentence-transformers>=3.0  # Embeddings (NB02, 04–07)
transformers>=4.40   # Model loading
faiss-cpu            # Retrieval (NB06–07)
bertopic>=0.17       # Topic modeling (NB04)
setfit>=1.1          # Few-shot (NB05)
unsloth, trl         # Fine-tuning (NB09)
evaluate             # Metrics
```

---

## 9. Implementation Order

### Phase 0 — Immediate: Project specification
1. **Create `/Users/roman/Desktop/UNISTRA-NLP-2026/CLAUDE.md`** with the full workshop specification (this plan) as the persistent source of truth for the project

### Phase 1 — Critical (Feb 5–6): Tuesday's material
2. Repository setup on GitHub (structure, .env.example, README)
2. **Webpage** (index.html + block panels) — Governance_presentations design
3. **NB01** (TF-IDF) — adapt from `M2_NLP_Supervised.ipynb`
4. **NB02** (Embeddings) — adapt from same source
5. **NB03** (LLM zero-shot + structured output) — adapt from Session 2 notebooks (swap Gemini/Ollama for Groq)
6. **NB04** (BERTopic) — adapt from `M2_NLP_TopicModelBert.ipynb`, update API calls

### Phase 2 — High Priority (Feb 7–8): Wednesday + Thursday AM
7. **NB05** (SetFit) — new, green claims dataset
8. **NB06** (FAISS) — new, multilingual retrieval
9. **NB07** (Reranking) — new, builds on NB06
10. **Kickoff presentation** (Beamer) — figures + compilation

### Phase 3 — Finalize (Feb 9): Thursday PM material
11. **NB08** (Distillation) — new
12. **NB09** (Fine-tuning Qwen3) — new
13. **NB10** (Evaluation) — new
14. **NB11** (Annotation/IRR) — new, shorter notebook
15. Test all Colab links, final webpage updates

### Phase 4 — During Workshop
16. Agent maintains webpage: update panels, post announcements, add links as sessions complete

---

## 10. Key Files to Reuse

| Source File | What to Extract |
|-------------|----------------|
| `old_notebooks/NLP_intro 2/M2_NLP_Supervised.ipynb` | TF-IDF pipeline, SBERT pipeline, evaluation code, dataset loading (dk_posts) |
| `old_notebooks/NLP_intro 2/M2_NLP_intro.ipynb` | "5 eras" narrative for kickoff presentation, similarity heatmap visualizations |
| `old_notebooks/NLP_intro 2/M2_NLP_TopicModelBert.ipynb` | BERTopic pipeline, topic visualization code |
| `old_notebooks/NLP_intro 2/dk_posts_synth_en_processed.json` | Demo dataset (copy to `data/`) |
| `old_notebooks/Session2_notebooks_data 2/m2_nlp_llmstructure.ipynb` | **Primary source for NB03/NB08**: Pydantic schema-enforced JSON, `extract_with_retry()`, batch processing, cost analysis, 3 real-world articles (AI scaling, VC funding, Danish policy) |
| `old_notebooks/Session2_notebooks_data 2/m2_nlp_ollama.ipynb` | **Primary source for NB03/NB09**: Ollama server management, zero-shot → few-shot classification with Pydantic Literal types, `classify_fast()` batch function, multilingual (EN+DA), sklearn evaluation |
| `old_notebooks/Session2_notebooks_data 2/dk_posts_synth.json` | Danish language version of demo dataset (for multilingual exercises) |
| `UNISTRA25/LLM_structured_output_AAUBS.ipynb` | Additional structured output patterns (Kluster/Together API variant) |
| `UNISTRA25/LLM_content_analysis_AAUBS-2.ipynb` | Additional Ollama deployment pattern |
| `UNISTRA25/02a_bertopic_contraversiers.ipynb` | Advanced BERTopic + LLM topic naming for NB04 |
| `/Users/roman/Desktop/Governance_presentations/index.html` | Webpage design (CSS, layout, panel system, JS) |
| 2024 repo: `UNISTRA-02-FewShot-classifier-argilla-setfit.ipynb` | SetFit + Argilla pattern for NB05/NB11 |
| 2024 repo: `UNISTRA-07-LLMLabelSynthesis.ipynb` | Distillation pattern for NB08 |

---

## 11. Verification

- [ ] Each notebook runs end-to-end in Google Colab (T4 runtime)
- [ ] Groq free-tier API calls work (test with `llama-3.1-8b-instant`)
- [ ] Together.AI fallback works
- [ ] Ollama installs and runs in Colab
- [ ] All Colab links resolve (repo must be public)
- [ ] Webpage renders on desktop + mobile
- [ ] Data files download correctly from GitHub
- [ ] Kickoff Beamer PDF compiles and looks correct
- [ ] `.env.example` lists all required keys
