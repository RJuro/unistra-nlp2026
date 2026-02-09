# Trademark × EU AI Act: Strategy for NB08 + NB09

## The Dataset

**Source:** `data/trademarks/euipo_tm_data.csv`
- 54,051 EUIPO digital product trademark filings (2015–2020)
- Fields: `owner_id`, `owner_name`, `ApplicationNumber`, `full_description`
- Descriptions are standardized goods/services terms accepted by EUIPO examiners
- Avg description length: 679 chars

## The Task

Classify each trademark's **EU AI Act risk tier** based on its goods/services description.

### Risk Tiers (5 classes)

| Tier | Definition | Examples in data | Est. count |
|------|-----------|-----------------|------------|
| **Unacceptable** | Social scoring, real-time biometric mass surveillance, manipulation | Very rare — need synthetic boost | ~50 synthetic |
| **High-risk** | Biometric ID, hiring tools, credit scoring, law enforcement, medical devices | 553 biometric, 423 facial recognition | ~800–1,200 |
| **Limited** | Chatbots, emotion detection, deepfake generation | 92 chatbot, some VR/voice | ~200–400 |
| **Minimal** | Spam filters, game AI, recommendation engines, search tools | Bulk of AI-adjacent filings | ~2,000–2,500 |
| **Not AI** | Hardware, cables, standard software, clothing | 93% of dataset | ~50,000 |

### AI-Adjacent Trademarks Already in Dataset

Total with any AI term: **3,765 / 54,051 (7.0%)**

Key term frequencies:
- `artificial intelligence`: 2,114
- `robot`: 1,264
- `machine learning`: 1,062
- `biometric`: 553
- `facial recognition`: 423
- `autonomous`: 282
- `data mining`: 167
- `predictive`: 107
- `voice recognition`: 106
- `virtual assistant`: 99
- `chatbot`: 92
- `image recognition`: 89
- `speech recognition`: 79
- `natural language`: 44
- `deep learning`: 9
- `computer vision`: 7

## Bootstrap Strategy

### Step 1: LLM-label the real data (NB08)

Use Groq (llama-3.1-8b or qwen3-32b) to classify a sample of ~2,000 trademarks:
- ~500 from the AI-adjacent subset (keyword-filtered)
- ~500 random from the full dataset (mostly "not_ai")
- Structured output with Pydantic:

```python
class AIActAssessment(BaseModel):
    is_ai_related: bool
    risk_tier: Literal["unacceptable", "high", "limited", "minimal", "not_ai"]
    risk_rationale: str
    ai_capabilities: list[str]  # e.g. ["facial recognition", "biometric identification"]
    confidence: float  # 0-1
```

### Step 2: Generate synthetic high-risk examples (NB08)

The "unacceptable" and some "high-risk" categories are underrepresented in real 2015–2020 filings. Use LLM to generate ~500 synthetic trademark descriptions in exact EUIPO style:

**Unacceptable tier targets:**
- Social credit scoring systems
- Real-time biometric surveillance in public spaces
- Subliminal manipulation software
- Emotion exploitation for vulnerable groups

**High-risk tier targets:**
- Automated hiring/recruitment screening
- Creditworthiness assessment
- Predictive policing / crime forecasting
- Autonomous vehicle control systems
- Medical AI diagnostic devices
- Educational assessment / student scoring

**Prompt pattern:**
```
Generate a realistic EUIPO trademark goods/services description for a {category} product.
Use the exact style of EUIPO standardized terms. Here are 3 real examples for reference:
{real_examples}
```

### Step 3: Merge + confidence filter + distill (NB08)

1. Combine real labeled + synthetic labeled data (~2,500 examples)
2. Confidence filter: keep only LLM labels with confidence > 0.7
3. Class balance: ensure minimum ~100 examples per tier
4. Train sklearn classifier (LogisticRegression on SBERT embeddings)
5. Compare to LLM zero-shot on held-out set

### Step 4: Fine-tune Qwen3-4B (NB09)

1. Format as instruction pairs:
   - Input: "Classify this EUIPO trademark under the EU AI Act risk framework: {description}"
   - Output: structured JSON with tier + rationale
2. LoRA fine-tune with Unsloth
3. Evaluate vs zero-shot (NB03 style) and vs distilled sklearn (NB08)
4. Export to GGUF/Ollama as bonus

## Analytics / Discussion Points

After classification, students can explore:
- **Temporal trends:** Did AI-related filings increase 2015→2020? (spoiler: yes, massively)
- **Geographic patterns:** Which countries file the most high-risk AI trademarks?
- **Company strategies:** Are big tech companies filing more in high-risk or minimal categories?
- **Pre/post GDPR:** Did filing patterns change after May 2018?
- **The bootstrap question:** How reliable are LLM-generated labels vs human annotation? (connects to NB11)

## Why This Works Pedagogically

1. **Real data** — actual IP filings, not toy examples
2. **Real regulation** — EU AI Act is live law, students at Strasbourg will care
3. **Natural class imbalance** — teaches practical ML (not everything is balanced MNIST)
4. **Synthetic data generation** — a real technique, not a hack
5. **Full pipeline** — LLM labeling → filtering → distillation → fine-tuning → evaluation
6. **Connects to lecture** — ties back to patent/innovation slides and social science framing
7. **Non-trivial** — LLM must understand both technology AND regulation to classify correctly
