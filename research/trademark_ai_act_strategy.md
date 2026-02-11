# Trademark × EU AI Act: Strategy for NB08 + NB09

## The Dataset

**Source:** `data/trademarks/euipo_tm_data.csv`
- 54,051 EUIPO digital product trademark filings (2015–2020)
- Fields: `owner_id`, `owner_name`, `ApplicationNumber`, `full_description`
- Descriptions are standardized goods/services terms accepted by EUIPO examiners
- Avg description length: 679 chars

## The Task

Two complementary tasks that showcase the full power of LLM structured output:

### Task A — Classification (single label)
Classify each trademark's **EU AI Act risk tier** based on its goods/services description.

### Task B — Structured Extraction (the real payoff)
Extract a rich regulatory assessment from each trademark description — not just a label, but structured fields that could feed directly into a compliance database or policy analysis pipeline.

**This is the pedagogical point**: classification is just `Literal["a", "b", "c"]`. Real-world NLP needs to *extract structured knowledge* — multiple fields, lists, nested reasoning. The fine-tuned model should learn to do this, not just pick a label.

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

---

## Structured Output Schema

### The Pydantic models (used by both teacher and student)

```python
from pydantic import BaseModel, Field
from typing import Literal

class AIActAssessment(BaseModel):
    """EU AI Act regulatory assessment of a trademark filing."""
    is_ai_related: bool = Field(description="Whether the trademark covers AI-related goods/services")
    risk_tier: Literal["unacceptable", "high", "limited", "minimal", "not_ai"] = Field(description="EU AI Act risk classification")
    confidence: float = Field(ge=0, le=1, description="Model confidence in the risk tier assignment")
    ai_capabilities: list[str] = Field(default_factory=list, description="Specific AI capabilities mentioned, e.g. ['facial recognition', 'predictive analytics']")
    target_sectors: list[str] = Field(default_factory=list, description="Application domains, e.g. ['healthcare', 'law enforcement', 'finance']")
    risk_rationale: str = Field(description="1-2 sentence explanation of why this tier was assigned")
```

### Why this schema matters pedagogically

| Field | What it teaches |
|-------|----------------|
| `risk_tier` | Basic classification — `Literal` types, constrained output |
| `ai_capabilities` | List extraction — variable-length, model must parse description terms |
| `target_sectors` | Domain reasoning — LLM must *infer* sectors from goods/services language |
| `risk_rationale` | Free-text generation — must be consistent with the classification |

Students see that a fine-tuned 4B model can learn to produce this *entire structure* from a trademark description — not just a label.

---

## Teacher Model

**`moonshotai/kimi-k2-instruct-0905`** via Groq API

- 1T total params, 32B active (MoE) — massive teacher for knowledge distillation
- Available on Groq free tier (same OpenAI-compatible API pattern)
- 262K context window
- Strong at structured output and regulatory reasoning

```python
client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
TEACHER_MODEL = "moonshotai/kimi-k2-instruct-0905"
```

---

## Bootstrap Strategy (NB08)

### Scale: ~200 real + ~200 synthetic = ~400 total

This is deliberately small — enough for LoRA fine-tuning, realistic for a student exercise, and avoids burning through Groq free tier limits.

### Step 1: LLM-label real trademarks (~200 examples)

Sample ~200 trademarks with stratified selection:
- ~100 from the AI-adjacent subset (keyword-filtered: any of the AI terms above)
- ~100 random from the full dataset (mostly "not_ai", some surprises)

For each trademark, the teacher model produces the full `AIActAssessment` schema:

```python
system_prompt = """You are an EU AI Act compliance analyst. Given a EUIPO trademark
goods/services description, produce a structured regulatory assessment.

The EU AI Act defines these risk tiers:
- unacceptable: Social scoring, real-time biometric mass surveillance, subliminal manipulation
- high: Biometric identification, hiring/recruitment tools, credit scoring, law enforcement,
  medical devices, critical infrastructure, education assessment
- limited: Chatbots, emotion detection, deepfake generation (transparency obligations only)
- minimal: Spam filters, game AI, recommendation engines, search tools (no obligations)
- not_ai: Products with no AI component

Respond with valid JSON matching the AIActAssessment schema."""

user_prompt = """Assess this EUIPO trademark under the EU AI Act:

Owner: {owner_name}
Application: {application_number}
Goods/Services: {full_description}"""
```

### Step 2: Generate synthetic examples (~200 examples)

The "unacceptable" and some "high-risk" categories are underrepresented in real 2015–2020 filings. Use kimi-k2 to generate ~200 synthetic trademark descriptions in exact EUIPO style, **then label them with the full schema**:

Target distribution for synthetic:
- ~40 unacceptable (social scoring, mass surveillance, manipulation, emotion exploitation)
- ~60 high-risk (hiring tools, credit scoring, medical AI, law enforcement, education)
- ~40 limited (chatbots, emotion detection, deepfakes)
- ~40 minimal (game AI, recommendations, search)
- ~20 not_ai (edge cases that look AI-adjacent but aren't)

**Two-step synthetic generation:**

```python
# Step 2a: Generate the trademark description
gen_prompt = """Generate a realistic EUIPO trademark goods/services description for a
{category} product. Use the exact style of EUIPO standardized terms — semicolon-separated
goods/services, formal register, no marketing language.

Category: {subcategory}
Here are 3 real EUIPO descriptions for reference style:
{real_examples}

Generate ONLY the goods/services description, nothing else."""

# Step 2b: Label with full schema (same prompt as Step 1)
# This ensures the synthetic data has the same rich structure
```

### Step 3: Merge + filter + format for fine-tuning (NB08 → NB09 handoff)

1. Combine real labeled + synthetic labeled data (~400 examples)
2. Confidence filter: keep only teacher labels with confidence > 0.7
3. Validate: drop any examples where JSON doesn't parse or fields are empty
4. Class balance check: ensure minimum ~30 examples per tier
5. Format as conversation pairs for Qwen3-instruct chat template
6. Save as HuggingFace Dataset or JSON for NB09

**Distillation comparison** (stays in NB08):
- Train sklearn classifier (LogisticRegression on SBERT embeddings) on just the `risk_tier` field
- Compare sklearn accuracy vs teacher LLM zero-shot on a held-out set of ~50 real trademarks
- Show: even basic distillation (embeddings + logistic regression) beats zero-shot for classification
- But: sklearn can't produce the full structured extraction — that's why we need NB09

---

## Fine-tuning Strategy (NB09)

### Model + Template (aligned with Unsloth reference notebook)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=2048,    # Trademark descriptions are short, 2048 is plenty
    load_in_4bit=True,      # Fits free Colab T4 (14.7 GB VRAM)
)
```

### LoRA Configuration

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### Data Format (chatml / qwen3-instruct)

Each example becomes a conversation pair:

```python
{
    "conversations": [
        {
            "role": "user",
            "content": "Assess this EUIPO trademark under the EU AI Act risk framework:\n\nOwner: CLEARVIEW AI INC.\nApplication: 18456789\nGoods/Services: facial recognition software; biometric identification systems; software for law enforcement agencies; real-time surveillance camera software; image matching databases"
        },
        {
            "role": "assistant",
            "content": '{"is_ai_related": true, "risk_tier": "high", "confidence": 0.95, "ai_capabilities": ["facial recognition", "biometric identification", "real-time surveillance", "image matching"], "target_sectors": ["law enforcement", "public security"], "risk_rationale": "Covers real-time biometric identification and facial recognition for law enforcement, falling under Annex III high-risk categories."}'
        }
    ]
}
```

Applied via the Unsloth template pattern:

```python
from unsloth.chat_templates import get_chat_template, train_on_responses_only

tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=False
    ) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Training Configuration

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,  # ~50 held-out examples
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=5,
        num_train_epochs=3,             # ~400 examples × 3 = 1,200 steps
        max_steps=None,                 # Let it run full epochs
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

# Only train on assistant responses (not the user prompts)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer.train()
```

### Evaluation (the 3-way comparison)

| Method | Classification | Full Extraction | Training Data |
|--------|---------------|-----------------|---------------|
| **kimi-k2 zero-shot** (teacher) | Baseline accuracy | Full schema | None |
| **sklearn on SBERT** (NB08 distill) | Competitive | Cannot do this | ~400 labeled |
| **Qwen3-4B LoRA** (NB09) | Compare to above | Full schema (!) | ~400 labeled |

Evaluation metrics:
- **Classification**: accuracy, macro F1, per-tier precision/recall on held-out ~50 real trademarks
- **Extraction quality**: field completeness (% non-empty), JSON validity rate, qualitative review of 10 examples
- **Speed**: tokens/sec for teacher vs fine-tuned model (the 4B model is 30× smaller → much faster)
- **Cost**: teacher API cost vs free local inference

### Export (bonus section)

```python
# Save to GGUF for Ollama
model.save_pretrained_gguf("trademark_classifier", tokenizer, quantization_method="q4_k_m")
```

Then serve locally:
```bash
ollama create trademark-aiact -f Modelfile
ollama run trademark-aiact "Assess this EUIPO trademark..."
```

---

## Analytics / Discussion Points

After classification, students can explore:
- **Temporal trends:** Did AI-related filings increase 2015→2020? (spoiler: yes, massively)
- **Geographic patterns:** Which countries file the most high-risk AI trademarks?
- **Company strategies:** Are big tech companies filing more in high-risk or minimal categories?
- **Pre/post GDPR:** Did filing patterns change after May 2018?
- **Extraction depth:** What AI capabilities appear most frequently across the corpus? (uses the `ai_capabilities` list field)
- **Sector mapping:** Which sectors are most represented in high-risk filings? (uses `target_sectors`)
- **The bootstrap question:** How reliable are LLM-generated labels vs human annotation? (connects to NB11)
- **Small vs large:** Where does the fine-tuned 4B model agree/disagree with the kimi-k2 teacher?

---

## Why This Works Pedagogically

1. **Real data** — actual IP filings, not toy examples
2. **Real regulation** — EU AI Act is live law, students at Strasbourg will care
3. **Natural class imbalance** — teaches practical ML (not everything is balanced MNIST)
4. **Synthetic data generation** — a real technique, not a hack
5. **Beyond classification** — structured extraction is what real NLP jobs require; classification is just one field
6. **Full pipeline** — LLM labeling → filtering → distillation → fine-tuning → evaluation
7. **Connects to lecture** — ties back to patent/innovation slides and social science framing
8. **Non-trivial** — LLM must understand both technology AND regulation to extract correctly
9. **Tractable scale** — 400 examples, ~15 min of API calls, ~20 min of fine-tuning on T4
10. **The distillation story is clear** — 1T-param teacher → 4B student, full structured output preserved
