# Applied NLP in 2026: A comprehensive lecture research brief

**This document provides the complete research foundation for a 45-minute Applied NLP lecture aimed at master's students in economics and social sciences.** It covers foundational NLP concepts with key references, the full 2025–2026 state of the art in language models, practical AI engineering patterns, multimodal AI, and — critically for this audience — how NLP methods are transforming social science research. Every section includes specific papers, authors, dates, and resources suitable for both slide preparation and recommended reading lists.

---

## PART 1: FOUNDATIONAL CONCEPTS

---

## 1. TF-IDF remains the quiet workhorse of text retrieval

**TF-IDF (Term Frequency–Inverse Document Frequency)** quantifies how important a word is to a document relative to a corpus. Despite being over 50 years old, it remains foundational in production search systems, RAG pipelines, and social science text analysis.

**Classic references.** Karen Spärck Jones introduced the IDF concept in her landmark 1972 paper "A Statistical Interpretation of Term Specificity and Its Application in Retrieval" (*Journal of Documentation*, 28(1): 11–21), arguing that matches on rarer, more specific terms carry greater value. Gerard Salton formalized the vector space model at Cornell's SMART project: see Salton, Wong & Yang (1975), "A Vector Space Model for Automatic Indexing" (*Communications of the ACM*, 18(11): 613–620), and Salton & Buckley (1988), "Term-weighting approaches in automatic retrieval" (*Information Processing and Management*).

**The information theory connection** is pedagogically powerful for an economics audience. IDF = log(N/df) is mathematically equivalent to the **self-information (surprisal)** of encountering a term — rare terms carry more information. Aizawa (2003), "An information-theoretic perspective of tf–idf measures" (*Information Processing and Management*, 39(1): 45–65), demonstrates that TF-IDF can be expressed as "probability-weighted amount of information." This connects directly to Shannon's framework, making IDF intuitive: a term appearing in every document (like "the") has zero information content; a term appearing in 1% of documents carries high surprise.

**BM25 in production (2025).** TF-IDF's descendant **BM25** (Okapi BM25) is the default ranking algorithm in Elasticsearch, Apache Solr, and Apache Lucene. BM25 improves on raw TF-IDF with term frequency saturation (diminishing returns for repeated terms via TF/(TF + k₁)) and document length normalization. In modern AI search pipelines, BM25 serves as the "first-stage retriever" alongside vector search — what practitioners call **hybrid search**. BM25 excels at exact matches (product codes, legal clauses, error messages), while vector search handles semantic similarity. A new algorithm called **BMX** (arXiv:2408.06643, 2024) combines entropy-weighted similarity with TF-IDF, outperforming BM25 on the BEIR benchmark.

**Best pedagogical resources:** "Understanding TF-IDF and BM-25" at kmwllc.com builds BM25 from TF-IDF step by step; "TF-IDF and BM25 for RAG — A Complete Guide" at ai-bites.net includes toy corpus examples and Python code.

---

## 2. Word embeddings encode meaning as geometry

Word embeddings represent words as dense vectors in continuous space, where geometric relationships encode semantic relationships. They remain widely used in social science research even as computer science has moved to transformer-based methods.

**Key original papers.** Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space" (arXiv:1301.3781) introduced **Word2Vec** with Skip-gram and CBOW architectures. Their companion paper "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013) added negative sampling and phrase-level embeddings. Pennington, Socher & Manning (2014), "GloVe: Global Vectors for Word Representation" (EMNLP 2014) combined global matrix factorization with local context windows (https://nlp.stanford.edu/projects/glove/). Bojanowski et al. (2017), "Enriching Word Vectors with Subword Information" (*TACL*, 5: 135–146) introduced **FastText**, which handles morphology through character n-grams and can produce embeddings for out-of-vocabulary words.

**The canonical analogy** is **king − man + woman ≈ queen**: vector arithmetic captures that gender, tense, geography, and other relationships are encoded as consistent directional offsets in embedding space. For a social science audience, the GPS analogy works well: words are assigned coordinates in "meaning-space" where proximity equals similarity, just as cities have GPS coordinates where Paris is closer to Lyon than to Tokyo.

**Bias in embeddings** is highly relevant for social scientists. Bolukbasi et al. (2016), "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings" (NeurIPS) demonstrated systematic gender stereotypes in Word2Vec trained on Google News. Caliskan, Bryson & Narayanan (2017), "Semantics derived automatically from language corpora contain human-like biases" (*Science*, 356: 183–186) introduced the Word Embedding Association Test (WEAT). Gonen & Goldberg (2019) showed debiasing methods are superficial ("Lipstick on a Pig"). Recent work continues: a 2025 systematic review in *Applied Intelligence* covers bias detection methods across languages, and Chuthamsatid et al. (2026) examine whether LLM-based embeddings from the MTEB leaderboard still exhibit biases.

**Word embeddings in social science (2023–2025)** — specific papers for the lecture:

- A comprehensive survey, "Word embedding for social sciences: an interdisciplinary survey" (*PeerJ Computer Science*, 2024, DOI: 10.7717/peerj-cs.2562), reviews 2013–2024 literature across sociology, economics, and political science.
- "Meaning in Hyperspace: Word Embeddings as Tools for Cultural Measurement" (*Annual Review of Sociology*, 2025) distinguishes "context spaces" from "concept spaces" — essential reading for methodological understanding.
- Aceves & Evans (2024), "Mobilizing Conceptual Spaces" (*Organization Science*) uses Word2Vec on patent abstracts (1976–2019) to represent the conceptual space of technological innovation.
- Kozlowski, Taddy & Evans (2019), "The Geometry of Culture" (*American Sociological Review*, 84(5): 905–949) remains seminal for using embeddings to measure cultural dimensions like class, gender, and race.
- Arseniev-Koehler (2024), "Theoretical Foundations and Limits of Word Embeddings" (*Sociological Methods & Research*) provides crucial methodological grounding.
- Mostafavi & Porter (2025), "Contextual Embeddings in Sociological Research" (*Sociological Methodology*) extends the approach to BERT-era contextual embeddings.

---

## 3. Topic modeling evolved from LDA to BERTopic

**LDA (Latent Dirichlet Allocation)** by Blei, Ng & Jordan (2003, *JMLR*, 3: 993–1022) is the foundational generative probabilistic model: documents are random mixtures over latent topics, each topic a distribution over words. Blei's 2012 survey "Probabilistic Topic Models" (*Communications of the ACM*, 55(4): 77–84) remains an excellent illustrated introduction. **NMF (Non-negative Matrix Factorization)**, from Lee & Seung (1999/2000, NeurIPS), provides a deterministic, often faster alternative that decomposes a document-term matrix into topic-term and document-topic matrices.

**BERTopic** (Grootendorst, 2022, arXiv:2203.05794) has become the dominant modern topic modeling approach. Its architecture is a four-step modular pipeline: (1) **sentence-transformers** generate document embeddings, (2) **UMAP** reduces dimensionality while preserving local structure, (3) **HDBSCAN** clusters documents and automatically determines topic count, and (4) **c-TF-IDF** (class-based TF-IDF) creates interpretable topic representations by comparing term frequencies within clusters against the corpus. BERTopic supports LLM-based topic labeling (using GPT-4 or Claude to name topics), multilingual analysis across **50+ languages**, dynamic and hierarchical topic modeling, and is fully modular — any component can be swapped. Performance benchmarks show BERTopic consistently achieves higher coherence than LDA and NMF; on Hindi short texts, BERTopic's coherence (Cv) is nearly double LDA's (**0.76 vs. 0.38**). One limitation is that HDBSCAN can assign a high proportion of short texts as outliers. The 2024–2025 updates (v0.17+) added multi-GPU UMAP and lightweight inference via Model2Vec.

**Documentation and tutorials:** https://maartengr.github.io/BERTopic/ (official), with excellent quickstart guides and social media analysis notebooks.

---

## 4. Sentence embeddings power modern similarity search

The evolution from **Doc2Vec** (Le & Mikolov, 2014, ICML) to **Sentence-BERT** (Reimers & Gurevych, 2019, EMNLP, arXiv:1908.10084) to today's models represents a major advance. SBERT's key innovation was using siamese BERT networks to produce fixed-size sentence embeddings comparable via cosine similarity, reducing the time to find the most similar sentence pair from **65 hours** (BERT cross-encoder) to **~5 seconds**.

**The MTEB (Massive Text Embedding Benchmark)** leaderboard (https://huggingface.co/spaces/mteb/leaderboard) evaluates models across classification, clustering, retrieval, and semantic similarity tasks. As of late 2025, the top models include **Cohere embed-v4** (score ~65.2, 1024 dims, $0.10/M tokens), **OpenAI text-embedding-3-large** (~64.6, 3072 dims), **NV-Embed** (NVIDIA, ~69.32 with latent attention layer), and **Qwen3 Embedding** (8B, Apache 2.0). For budget-conscious social scientists, **all-MiniLM-L6-v2** (384 dims, free, fast) or **BGE-M3** (1024 dims, open-source) remain excellent practical choices.

**Key architectural innovations** include **Matryoshka embeddings** (Kusupati et al., 2022, NeurIPS, arXiv:2205.13147) — like Russian nesting dolls, the first *m* dimensions of a *d*-dimensional embedding are independently useful, letting users choose embedding size at inference time with minimal accuracy loss. This is now adopted by OpenAI and Google Gemini embedding APIs. **ColBERT** (Khattab & Zaharia, 2020, arXiv:2004.12832) uses "late interaction" — retaining per-token embeddings and computing MaxSim between query and document tokens, offering a powerful balance of efficiency and accuracy for retrieval. **E5** (Wang et al., Microsoft) provides instruction-finetuned embeddings, while **GTE** (Alibaba/DAMO) builds on the Qwen family for strong multilingual performance. **Jina Embeddings v3** supports multiple languages, long context, and Matryoshka dimensions.

---

## 5. Transformers are the architecture behind everything

**"Attention Is All You Need"** (Vaswani et al., 2017, NeurIPS, arXiv:1706.03762) introduced the transformer architecture, replacing recurrence with **self-attention**: each word attends to all other words, computing attention scores as softmax(QKᵀ/√d_k)V. Multi-head attention runs multiple "heads" in parallel, each learning different relationship patterns. Positional encodings (sine/cosine functions) supply sequence order information since transformers lack inherent sequential processing. The architecture's parallelizability enabled massive speedups over RNNs.

**Three architectural lineages** matter for the lecture:

| Architecture | Examples | Task Type | Key Property |
|---|---|---|---|
| **Encoder-only** | BERT (Devlin et al., 2019), RoBERTa | Classification, NER, similarity | Bidirectional attention; understands meaning |
| **Decoder-only** | GPT-2/3/4, LLaMA, Claude | Text generation, chat | Causal (left-to-right); generates token by token |
| **Encoder-Decoder** | T5 (Raffel et al., 2020), BART | Translation, summarization | Encoder understands input, decoder generates output |

For non-CS audiences, the **best pedagogical explanation** comes from Jay Alammar's "The Illustrated Transformer" (https://jalammar.github.io/illustrated-transformer/), used at Stanford, Harvard, MIT, Princeton, and CMU, now expanded into a book at LLM-book.com. **3Blue1Brown's** videos "Transformers, the tech behind LLMs" and "Attention in transformers, step-by-step" (https://www.3blue1brown.com/lessons/attention) are the clearest animated explanations available — Simon Willison called the attention video "by far the clearest explanation I've seen anywhere." **Lilian Weng's** blog posts "Attention? Attention!" and "The Transformer Family Version 2.0" (2023) at lilianweng.github.io provide deeper technical coverage. The **Transformer Explainer** from Georgia Tech (https://poloclub.github.io/transformer-explainer/) offers an interactive browser-based visualization of GPT-2.

**Key analogies for the lecture.** For attention: "Query is your search text, Key is the page title, Value is the page content" — you match your query against keys and retrieve the relevant values. For next-token prediction: Jay Alammar's "smartphone keyboard on steroids" analogy — GPT is the keyboard autocomplete feature scaled massively, requiring deep world knowledge. And a memorable counter to dismissiveness: "Saying LLMs 'just predict the next word' is like saying a cathedral is 'just a pile of stones.'"

---

## PART 2: 2025/2026 STATE OF THE ART

---

## 6. The LLM landscape has never been more competitive

As of February 2026, the large language model ecosystem is defined by fierce competition across proprietary and open-weight providers, plummeting costs, and rapidly saturating benchmarks.

**OpenAI** progressed from **GPT-4o** (May 2024, 128K context, $2.50/$10 per M tokens — being retired Feb 13, 2026) through **GPT-4.1** (April 2025, 1M context), to the current flagship **GPT-5/5.1/5.2** (August–December 2025). GPT-5.2 offers a **400K context window** and **128K output tokens** at $1.75/$14 per M tokens, scoring **100% on AIME 2025** and **35.4% on HLE**. The reasoning line includes **o1** (September 2024, internal chain-of-thought), **o3** (April 2025, tool use, **87.5% on ARC-AGI**), and **o3-mini** ($1.10/$4.40 per M tokens).

**Anthropic** released **Claude 3.5 Sonnet** (June 2024, updated October 2024 with computer-use), **Claude 3.7 Sonnet** (February 2025, first hybrid reasoning model), **Claude 4 Opus/Sonnet** (May 2025), and **Claude 4.5 Sonnet** (September 2025, **77.2% on SWE-bench Verified**, best coding model). Claude models offer **200K context** standard with **1M beta** available via API. Claude Opus 4.5 ($15/$75 per M tokens) and Haiku 4.5 ($0.25/$1.25) round out the family.

**Google DeepMind** launched **Gemini 2.0 Flash** (December 2024/February 2025, 1M context, native tool use) and **Gemini 2.0 Pro** (**2M context window** — among the largest). The current flagship **Gemini 3 Pro** (November 2025) was the first model to break **1500 Elo on LMArena** (1501), scoring **91.9% on GPQA Diamond** and a leading **37.2% on HLE**.

**Meta** released **Llama 3.3** (December 2024, 70B, 128K context) and **Llama 4** (April 2025) in MoE configurations: **Scout** (17B active, 109B total, **10M token context** — the largest of any model) and **Maverick** (17B active, 400B total, 1M context). All Llama 4 under community license, free for <700M monthly users.

**Microsoft Phi-4** (late 2024/early 2025): **16B parameters**, MIT license, trained on synthetic data, scoring **82.6% on HumanEval** — competitive with much larger models.

**Mistral Large 3** (2025): 675B total MoE parameters, delivering ~92% of GPT-5.2 performance at ~15% of the price. **Mistral Small 3** (24B, Apache 2.0) runs on a single GPU.

**Benchmark saturation and new frontiers.** MMLU is now saturated (90%+ for frontier models). The differentiating benchmarks are **GPQA Diamond** (PhD-level science), **AIME** (competitive math), **SWE-bench** (real-world software engineering), and especially **Humanity's Last Exam (HLE)**.

**Humanity's Last Exam (HLE)** was created by the **Center for AI Safety (CAIS)**, led by Dan Hendrycks, in collaboration with **Scale AI**. Nearly **1,000 experts from 500+ institutions across 50 countries** contributed 2,500 questions across 100+ academic subjects, specifically designed to stump AI models. Published in *Nature* (2025, DOI: 10.1038/s41586-025-09962-4) and on arXiv (2501.14249). When released in January 2025, top models scored in the **single digits**. By early 2026, scores have climbed to the mid-30s: **Gemini 3 Pro at 37.2%**, **GPT-5.2 at 35.4%**, vs. human experts at ~90%. Progress has been rapid but the benchmark remains far from saturated. Leaderboard: https://scale.com/leaderboard/humanitys_last_exam

**Context windows in early 2026:** 200K+ is offered by Claude and OpenAI o-series models. 1M+ tokens are available from Gemini (2M), GPT-4.1, Claude 4.5 (beta), Llama 4 Maverick, and Qwen 2.5-1M. **Llama 4 Scout's 10M token context** (~7,500 pages of text) is unprecedented. Google's needle-in-a-haystack tests show near-perfect retrieval in Gemini across text, video (10.5 hours), and audio (107 hours).

**Cost-performance as of early 2026:**

| Model | Input $/M | Output $/M | Key Strength |
|-------|-----------|------------|---|
| DeepSeek V3.2 | **$0.27** | **$0.42** | Best cost-performance ratio |
| DeepSeek R1 | $0.55 | $2.19 | Reasoning at 90–96% lower cost than o1 |
| GPT-4o Mini | $0.15 | $0.60 | Best Western budget option |
| Claude Sonnet 4.5 | $3.00 | $15.00 | Best coding model |
| GPT-5.2 | $1.75 | $14.00 | Frontier reasoning |
| Gemini 2.0 Flash | Low | Low | 1M context, free tier |

**DeepSeek V3.2 at $0.27/$0.42** is 10–30x cheaper than GPT-5.2 while offering competitive reasoning. Chinese AI models grew from ~1.2% of global usage (late 2024) to **~30% (end 2025)**, largely driven by cost advantages. Output tokens cost **3–8x more than input** across all providers.

---

## 7. Reasoning models represent a paradigm shift in inference

**Chain-of-thought prompting** (Wei et al., 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS, arXiv:2201.11903) was the foundational discovery: providing a few exemplars of intermediate reasoning steps in prompts enables LLMs to decompose complex problems. A critical finding was that **CoT is an emergent property of scale** — only appearing in models above ~100B parameters.

**OpenAI's o1** (September 2024) was the first commercial reasoning model, trained via reinforcement learning to generate an internal chain of thought before responding. It represented a paradigm shift: instead of scaling model size, **improve by spending more compute at inference time**. o1 scored **83.3% on AIME 2024** (vs. GPT-4o's 9.3%) and **75.7% on GPQA Diamond**. The theoretical foundation comes from Snell et al. (2024, arXiv:2408.03314), "Scaling LLM Test-Time Compute Optimally," which showed that a smaller model with more inference compute can outperform a **14x larger model** on problems with non-trivial success rates.

**OpenAI's o3** (announced December 2024, released April 2025) added autonomous tool use and achieved **96.7% on AIME 2024**, **87.7% on GPQA Diamond**, and a groundbreaking **87.5% on ARC-AGI** — near human-level (85%) and described by ARC Prize as "a genuine breakthrough, marking a qualitative shift in AI capabilities." On EpochAI Frontier Math, o3 scored **25.2%** while no other model exceeded 2%. However, high-compute ARC testing cost ~$34,400/task vs. ~$5 for a human.

**DeepSeek-R1** (January 20, 2025, arXiv:2501.12948, also published in *Nature*) took a radically different approach. Built on the 671B-parameter DeepSeek-V3 base, R1 was first trained as **R1-Zero** — using **pure reinforcement learning (GRPO algorithm) without any supervised fine-tuning**, a groundbreaking first. Using only rule-based rewards (accuracy + format), the model naturally developed self-verification, reflection, and "aha moments." The full R1 added cold-start SFT and a second RL stage. R1 matches o1 on most benchmarks (**79.8% AIME, 97.3% MATH-500, 90.8% MMLU**) at **90–96% lower cost** ($0.55/$2.19 vs. $15/$60), and is fully **open-source under MIT license**. The updated R1-0528 (May 2025) approaches o3 performance levels.

**Visible vs. hidden reasoning** is a key distinction. DeepSeek-R1 provides **fully transparent reasoning traces** via `<think>...</think>` tags. OpenAI's o1/o3 hide their internal chain-of-thought, citing safety and competitive advantage — a "loss of transparency" that concerned developers. Under competitive pressure from R1, OpenAI enhanced o3-mini's transparency in February 2025.

**Distillation from reasoning models** is a critical practical finding: DeepSeek released 6 distilled models (1.5B–70B based on Qwen2.5 and Llama3). The **32B distilled model outperforms OpenAI o1-mini** across all benchmarks tested, demonstrating that reasoning patterns can be effectively transferred to smaller models.

---

## 8. Chinese AI labs reshaped the global competitive landscape

**DeepSeek** (founded by Liang Wenfeng, backed by quantitative hedge fund High-Flyer) delivered two landmark releases. **DeepSeek-V3** (December 26, 2024) uses a **Mixture-of-Experts (MoE)** architecture with **671B total parameters but only 37B activated per token** (~5.5% utilization). Key innovations include Multi-head Latent Attention (MLA) for efficient KV cache, auxiliary-loss-free load balancing, multi-token prediction, and — critically — **FP8 mixed-precision training** validated at extreme scale, effectively doubling compute efficiency. Training cost: **~$5.6M** on 2,048 H800 GPUs for 14.8T tokens. For comparison, GPT-4's training was estimated at $50–100M. Engineers programmed directly in **PTX** (assembly-level GPU code) to optimize cross-chip communication on bandwidth-limited H800s. The updated **V3.2** (December 2025) introduced "DeepSeek Sparse Attention" with 50% compute reduction, priced at an extraordinarily aggressive **$0.27/$0.42 per M tokens**.

An important caveat from SemiAnalysis: the $5.6M covers only the final training run, not R&D, ablations, or hardware acquisition. DeepSeek reportedly has access to ~50,000 Hopper GPUs worth over **$500M in total investment**.

**Qwen (Alibaba)** has become the most-used open-source LLM family globally. **Qwen 2.5** (September 2024) spans 0.5B to 72B parameters under **Apache 2.0** license, with specialized coding and math variants. By September 2025, Qwen overtook Meta's Llama as the **most downloaded LLM family on Hugging Face**, with 40M+ downloads and 180,000+ derivative models. **Qwen 3** (April 2025) added MoE variants (235B/22B active) trained on 36T tokens in **119 languages** with hybrid thinking mode. Alibaba has open-sourced 200+ generative AI models.

**Other notable Chinese labs.** The "Six Tigers" (each $1B+ valuation) are **Zhipu AI** (Tsinghua spin-off, GLM-4.5 series), **MiniMax** (1M context specialist), **Baichuan** (Chinese language specialist), **Moonshot AI/Kimi** (20M+ users, Kimi K2 model), **StepFun**, and **01.AI** (by Kai-Fu Lee, Yi-Lightning ranked joint 3rd on LMSYS with training costs ~2% of xAI's).

**The geopolitical dimension** is essential context for this audience. US export controls banned A100/H100 chips for China in October 2022, then restricted H800s in October 2023. Chinese labs adapted through stockpiling pre-ban chips, assembly-level GPU optimization, and — most importantly — **algorithmic efficiency innovations** (MoE, FP8 training, multi-token prediction). The Brookings Institution concluded (February 2025): "Scarcity fosters innovation. As a direct result of U.S. controls, companies in China are creating new AI training approaches that use computing power very efficiently." When DeepSeek-R1 launched January 27, 2025, NVIDIA lost **>$600B in market cap in a single day**, and DeepSeek's app overtook ChatGPT as #1 on the Apple App Store. Stanford HAI (2025) reports Chinese developers now account for **17.1%** of all Hugging Face downloads (vs. US at 15.8%), and **63%** of all new fine-tuned models are based on Chinese base models.

---

## 9. AI engineering in 2025 has a mature toolkit

### Retrieval-Augmented Generation (RAG)

RAG (Lewis et al., 2020, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," arXiv:2005.11401) equips LLMs with external knowledge sources to ground answers. The field has evolved through three tiers: **Naive RAG** (query → embed → retrieve → stuff into prompt → generate), **Advanced RAG** (adds query rewriting, hybrid search, reranking, evaluation guardrails), and **Modular/Agentic RAG** (separates retriever, generator, and orchestration into independent modules with planning loops). Best chunking practice in 2025 is **semantic chunking** using lightweight models to identify topic transitions, with moderate chunk sizes (**200–300 words**). **Hybrid search** combining BM25 + vector search via Reciprocal Rank Fusion (RRF) consistently outperforms either method alone. Reranking with cross-encoders (Cohere Rerank, ColBERT) further improves retrieval quality. Major frameworks: **LangChain/LangGraph** (most widely used), **LlamaIndex** (RAG specialist), and **Haystack** (by deepset). Key reference: arXiv:2501.07391 (January 2025), "Enhancing Retrieval-Augmented Generation: A Study of Best Practices."

### Agents and tool use

The industry has shifted from "generative AI" to **"agentic AI."** LLMs output structured function calls (JSON) which the application executes, returning results for further reasoning. **MCP (Model Context Protocol)**, released by Anthropic in November 2024 as an open standard, standardizes how AI applications connect to external tools — described as **"USB-C for AI."** It solves the N×M integration problem: instead of custom connectors for every data source × every AI app, developers implement MCP once. Now hosted by The Linux Foundation and adopted by OpenAI, Google, Zed, Replit, and Sourcegraph. Major agent frameworks include LangGraph (graph-based stateful workflows), CrewAI (role-based multi-agent), AutoGen (Microsoft, conversation-based multi-agent), and smolagents (HuggingFace, lightweight). Google's **A2A (Agent-to-Agent)** protocol is emerging for multi-agent collaboration.

### Structured output and fine-tuning

**Structured output** from LLMs has become reliable. OpenAI's Structured Outputs (August 2024) guarantee JSON schema compliance using constrained decoding. The **instructor library** by Jason Liu (https://python.useinstructor.com/, 3M+ monthly downloads) patches LLM clients to accept Pydantic models as `response_model`, automatically validating and retrying. The **Outlines library** (https://github.com/dottxt-ai/outlines) works at the token-generation level using finite state machines to mask invalid tokens during generation.

For **fine-tuning**, the standard approach is **QLoRA** (Dettmers et al., 2023, arXiv:2305.14314) — combining LoRA's low-rank adaptation (Hu et al., 2021, arXiv:2106.09685) with 4-bit quantization, enabling fine-tuning of 70B models in <48GB VRAM. **Unsloth** (https://unsloth.ai/) delivers **2x faster** fine-tuning with **70% less VRAM** through hand-derived backpropagation and Triton kernels, with free Google Colab notebooks. The decision framework: try **prompting** first (zero cost), then **RAG** (for external knowledge), then **fine-tuning** (for behavior/tone/format customization).

**Distillation** — using large models to generate training data for small models — has proven highly effective. DeepSeek-R1's distilled Qwen-32B outperforms o1-mini. Legal caution: OpenAI and Anthropic terms prohibit using outputs for competitive model training; DeepSeek-R1 (MIT) and Qwen (Apache 2.0) explicitly permit distillation.

### Evaluation

**LLM-as-judge** uses GPT-4 or Claude to evaluate other LLM outputs, with G-Eval providing chain-of-thought scoring. **RAGAS** (https://github.com/vibrantlabsai/ragas) evaluates RAG pipelines across faithfulness, contextual relevancy, answer relevancy, and precision. **DeepEval** (https://github.com/confident-ai/deepeval) offers "pytest for LLMs" with 60+ built-in metrics. **Promptfoo** (https://github.com/promptfoo/promptfoo) enables YAML-based prompt testing. **W&B Weave** adds tracing and evaluation dashboards. Best practice: combine automated metrics + LLM-judge + human evaluation for critical cases.

---

## 10. Multimodal AI reached production maturity in 2025

**CLIP** (Radford et al., 2021, OpenAI, "Learning Transferable Visual Models From Natural Language Supervision") aligned text and image representations via contrastive learning on 400M image-text pairs — the foundational architecture behind Stable Diffusion, DALL-E, and zero-shot image classification. **SigLIP** (Zhai et al., ICCV 2023) replaced CLIP's softmax loss with simpler pairwise sigmoid loss, improving memory efficiency and small-batch performance. **SigLIP 2** (Google, February 2025) added captioning pretraining, self-supervised losses, and **80+ languages** — now the preferred vision encoder for many VLMs.

**Image generation** has matured rapidly. **FLUX** (Black Forest Labs, founded by the original Stable Diffusion creators) is the leading open-weight model — a **12B-parameter** rectified flow transformer with superior text rendering and photorealism. FLUX.2 (November 2025) ships in Pro, Flex, Dev, and Klein (Apache 2.0) variants. **GPT Image 1.5** (OpenAI, December 2025) leads the LM Arena (score 1264) with best-in-class text rendering and prompt following using an autoregressive (not diffusion) approach. **Midjourney V7** (April 2025) fixed chronic hand anatomy issues and excels at cinematic/artistic quality. **Ideogram 3.0** (March 2025) specializes in accurate text rendering within images. **Stable Diffusion 3.5** uses a Diffusion Transformer with flow matching but has fallen behind FLUX and proprietary models.

**Video generation** crossed from "impressive demos" to production viability. **Sora 2** (OpenAI, September 2025) generates up to **25-second videos with synchronized audio** (dialogue, sound effects, music), with a Disney partnership licensing 200+ characters. **Runway Gen-4/4.5** leads in character consistency across scenes with 4K output. **Kling 3.0** (Kuaishou, February 2026) offers native **4K at 60fps** with multi-shot storyboards. **Google Veo 3/3.1** leads in photorealism with integrated audio via YouTube training data. **WAN 2.6** (Alibaba) is notable as an open-source competitor.

**Vision-language models** have exploded. GPT-4o, Gemini, and Claude all handle image understanding natively. Open-source VLMs now competitive: **Qwen2.5-VL** (3B–72B, video input, 29 languages), **LLaMA 3.2 Vision** (11B–90B), **Gemma 3** (1B–27B, 140+ languages), and ultra-small models like **SmolVLM** (256M–2.2B, runs on iPhone).

**Audio/speech** has been transformed. **Whisper large-v3** (OpenAI, November 2023, 1.55B params) trained on 1M hours of audio delivers 10–20% error reduction over v2 across languages. The **Turbo** variant cut decoder layers from 32 to 4 for a **5.4x speedup** with minimal accuracy loss. OpenAI's newer **gpt-4o-transcribe** models (March 2025) outperform Whisper via GPT-4o architecture with RL training. **ElevenLabs Eleven v3** (June 2025) leads TTS quality with voice cloning, 32 languages, and emotional control tags. **NotebookLM Audio Overview** (Google) creates AI-generated podcast-style conversations from documents, now supporting **80+ languages** and multiple formats (Deep Dive, Brief, Critique, Debate) — a genuinely new content format. Real-time speech-to-speech models (OpenAI gpt-realtime, Qwen 2.5 Omni, Gemini Live) now eliminate the traditional ASR→LLM→TTS pipeline.

---

## PART 3: APPLIED SOCIAL SCIENCE

---

## 11. NLP is transforming how economists and social scientists work

### Embeddings for economic geography and innovation

The intersection of NLP with economic geography centers on measuring **technological relatedness** — how knowledge domains relate to each other. Balland, Boschma, Crespo & Rigby (2019), "Smart Specialization Policy in the EU" (*Regional Studies*, 53(9): 1252–1268) developed the analytical framework using co-occurrence measures in patent technology classes to guide EU Smart Specialization policy. Sam Arts and collaborators at KU Leuven have pioneered patent text analysis: Arts, Hou & Gomez (2021), "Natural Language Processing to Identify the Creation and Impact of New Technologies in Patent Text" (*Research Policy*, 50(2): 104144) used NLP to extract technical keywords and create new measures of technology creation and impact, with open code and data. Arts, Cassiman & Gomez (2018), "Text Matching to Measure Patent Similarity" (*Strategic Management Journal*, 39(1): 62–84) validated that text similarity outperforms bibliographic methods.

For **measuring innovation**, the landmark paper is Kelly, Papanikolaou, Seru & Taddy (2021), "Measuring Technological Innovation over the Long Run" (*AER: Insights*, 3(3): 303–320): a patent is "important" if textually distant from prior work but similar to subsequent work, using **TF-IDF text similarity**. This covers 1840–present and identified technological waves. A Scientometrics 2025 paper introduces **"semantic proximity"** using BERTopic and BERT similarity to map regional knowledge spaces, finding regions with higher semantic proximity to quantum science produce significantly more quantum-related publications.

For **patent analysis**, Hain, Jurowetzki et al. (2022), "A Text-Embedding-Based Approach to Measuring Patent-to-Patent Technological Similarity" (*Technological Forecasting and Social Change*, 177) uses Word2Vec embeddings with approximate nearest neighbor search. Bekamiri, Hain & Jurowetzki (2024), "PatentSBERTa" (*Technological Forecasting and Social Change*, 206) uses Sentence-BERT for patent distance and classification. A comprehensive survey, "Natural Language Processing in Patents" (arXiv:2403.04105, 2024), covers the full landscape and notes that LLMs remain "underdeveloped" in patent analysis compared to BERT-based approaches.

### Policy analysis, cultural analytics, and the text-as-data paradigm

**Text-as-data in economics** is anchored by two essential surveys: Gentzkow, Kelly & Taddy (2019), "Text as Data" (*Journal of Economic Literature*, 57(3): 535–574) canonized the approach, and Ash & Hansen (2023), "Text Algorithms in Economics" (*Annual Review of Economics*, 15: 659–688) is the first major economics survey to systematically cover embeddings and transformers. For **policy text analysis**, Rodriguez et al. (2024) demonstrated LLMs (ChatGPT, Llama 2) achieving F1 >93% for automated privacy policy analysis (*Computing*, 106: 3879–3903). Gandhi et al. (2024) apply NLP to analyze climate policies across countries.

In **cultural analytics**, Kozlowski, Taddy & Evans (2019), "The Geometry of Culture" (*American Sociological Review*) used embeddings to measure cultural dimensions. Hou & Huang (2025), "Natural Language Processing for Social Science Research: A Comprehensive Review" (*Chinese Journal of Sociology*/Sage) provides the latest comprehensive review. LLMs as "simulated economic agents" (Horton, 2023, NBER) — using LLMs to simulate human subjects ("Homo Silicus") for behavioral research — opens new methodological frontiers.

### The adoption gap between CS and social science

Social scientists are still widely adopting word embeddings and topic models while CS has moved to LLMs. This gap has structural causes: the **4-year diffusion timeline** between Gentzkow et al.'s 2019 canonization of bag-of-words methods and Ash & Hansen's 2023 introduction of embeddings/transformers mirrors historical patterns. Social science demands **interpretability and causal identification** that embeddings and topic models provide more readily than LLM outputs. **Computational barriers** persist — static embeddings run on laptops while BERT requires GPUs. BERTopic is bridging this gap, and LLMs-as-annotation-tools (Gilardi et al., 2023 showed ChatGPT outperforms crowd-workers for text annotation) provide a low-barrier entry point.

---

## 12. AI task exposure reshapes the economics of knowledge work

**The foundational framework** comes from Eloundou, Manning, Mishkin & Rock (2023/2024), "GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models" (*Science*, 384: 1306–1308, arXiv:2303.10130). Using O*NET task analysis with both human and GPT-4 annotators, they found **~80% of the US workforce** could have ≥10% of tasks affected, and ~19% could have ≥50% affected. Higher-income jobs face greater exposure. Felten, Raj & Seamans (2021/2023) constructed complementary AI occupational exposure indices mapping AI capabilities to O*NET abilities (*Strategic Management Journal*, 42(12): 2195–2211).

**Daron Acemoglu** (2024 Nobel laureate) provided the macroeconomic counterweight: "The Simple Macroeconomics of AI" (*Economic Policy*, 40(121): 13–58, January 2025; NBER WP 32487) uses task-based models to estimate AI's macro effect at **≤0.66% TFP increase over 10 years** — far more modest than Silicon Valley claims.

**Three landmark experimental studies** on LLM productivity are essential for the lecture:

**Dell'Acqua et al. (2023)** studied **758 BCG consultants** (HBS WP 24-013). For tasks within AI's capability frontier, consultants with GPT-4 were 25.1% faster with **40% higher quality**. For tasks outside the frontier, they performed **19 percentage points worse**. AI leveled the playing field: the lowest-performing consultants gained **43% improvement**. Two patterns emerged: "Centaurs" (divide tasks between human and AI) and "Cyborgs" (constant AI integration). A concerning finding: AI led to **more homogenized outputs**.

**Noy & Zhang (2023)** (*Science*, 381: 187–192) tested 453 professionals doing writing tasks. ChatGPT reduced time by **40%** and raised quality by **18%**, with the greatest benefits for lower-ability workers — compressing the productivity distribution.

**Brynjolfsson, Li & Raymond (2023/2025)** (*Quarterly Journal of Economics*, 140(2): 889–942) studied 5,172 customer support agents in a real deployment. Productivity increased **14–15%** overall, with **34% improvement for novice workers** and minimal impact on experts. The AI tool appeared to **disseminate tacit knowledge** of top performers to less experienced agents.

For the student audience: these findings suggest NLP tools will transform research workflows through automated text annotation at scale, LLM-assisted coding in R/Python/Stata, literature synthesis, and processing of previously inaccessible text data (historical archives, legal documents, policy texts). The key insight is that **AI amplifies rather than replaces** human expertise — but the "jagged frontier" means users must recognize which tasks fall within vs. outside AI capability boundaries.

---

## PART 4: PEDAGOGICAL RESOURCES

---

## 13. The best resources for teaching NLP to social scientists

**Core textbooks.** Grimmer, Roberts & Stewart (2022), *Text as Data: A New Framework for Machine Learning and the Social Sciences* (Princeton University Press) is the definitive textbook for this audience — organized around research tasks (representation, discovery, measurement, prediction, causal inference). Jurafsky & Martin, *Speech and Language Processing* (3rd edition draft, January 2025, **free online** at https://web.stanford.edu/~jurafsky/slp3/) is the comprehensive NLP reference. For LLM-era depth, Sebastian Raschka (2024), *Build a Large Language Model (From Scratch)* (Manning) walks through a GPT-2 implementation in PyTorch on a laptop, with free companion notebooks at https://github.com/rasbt/LLMs-from-scratch. Burkov's *The Hundred-Page Language Models Book* (https://thelmbook.com/) offers a compact ~100-page overview accessible to non-CS readers.

**Courses specifically for social scientists.** Stanford CS224C, "NLP for Computational Social Science" (Diyi Yang) directly bridges NLP methods and social science questions. JHU EN.601.472/672, "NLP for Computational Social Science" includes guest lectures from social scientists. The **SICSS** (Summer Institute in Computational Social Science) curriculum at https://sicss.io/curriculum provides free open-source materials (videos, slides, R code) with Day 3 covering automated text analysis including dictionary methods, topic modeling, and classification. **Machine Learning for Economists** (Itamar Caspi, Hebrew U.) at https://ml4econ.github.io integrates ML with econometric perspectives.

**Visual explanation resources** (ranked by accessibility for non-CS audiences):

- **Jay Alammar's Illustrated series** — The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/), The Illustrated BERT, The Illustrated Word2vec, and The Illustrated GPT-2. The gold standard for visual NLP explanations, now expanded into a book.
- **3Blue1Brown** — "Attention in transformers, step-by-step" (https://www.3blue1brown.com/lessons/attention) and "Transformers, the tech behind LLMs" provide the clearest animated explanations available.
- **Transformer Explainer** (Georgia Tech) — https://poloclub.github.io/transformer-explainer/ offers interactive browser-based GPT-2 visualization.
- **Hugging Face NLP/LLM Course** — https://huggingface.co/learn/llm-course — 12 chapters with code exercises, recently expanded to cover fine-tuning and reasoning models.

**Top analogies for the lecture** (tested with non-technical audiences):

- **Embeddings:** "Words as GPS coordinates in meaning-space — similar words are nearby, relationships are directions." The king−man+woman≈queen arithmetic shows that gender is a consistent direction.
- **Attention:** "Like a web search — Query is what you're looking for, Key is the title of each result, Value is the actual content. You match your query against relevant keys and pull the corresponding values."
- **What LLMs do:** "A smartphone keyboard autocomplete on steroids — next-word prediction scaled massively, which turns out to require deep world knowledge." Counter-analogy for skeptics: "Saying LLMs 'just predict the next word' is like saying a cathedral is 'just a pile of stones.'"
- **Transformers:** "An assembly line where each station (layer) adds more context and nuance. The word 'king' starts generic but, layer by layer, becomes 'a Scottish king who murdered the previous king, described in Shakespearean language.'"

---

## Conclusion: five themes for the lecture narrative

This research reveals five unifying themes that could structure the lecture's narrative arc. First, **the foundational methods haven't been replaced** — TF-IDF lives inside BM25 which powers every search engine and RAG pipeline; word embeddings remain the primary NLP tool in social science research. Understanding these basics is not just pedagogical but practical. Second, **the reasoning revolution** (o1, o3, DeepSeek-R1) represents a genuine paradigm shift from "make the model bigger" to "let the model think longer," with profound implications for how AI handles complex analytical tasks. Third, **the cost of intelligence is collapsing** — DeepSeek V3.2 at $0.27/M tokens delivers capabilities that cost 100x more two years ago, democratizing access for researchers with limited budgets. Fourth, **the toolkit for applied NLP has matured** from fragmented libraries into coherent patterns (RAG, agents, structured output, fine-tuning, evaluation) that social scientists can adopt systematically. Fifth, **the adoption gap between CS and social science is closing rapidly**, with BERTopic, LLM-based annotation, and structured output from language models providing accessible entry points for economists and social scientists who have specific, valuable domain problems that these tools can help solve.

The complete set of key URLs, papers, and resources compiled in this report should provide sufficient material for both the lecture slides and the recommended reading list for the workshop's hands-on notebook sessions.