# Figures Manifest for lecture/slides.tex

## Figures Generated Programmatically (TikZ in LaTeX)
These are drawn directly in the slides using TikZ — no external files needed:

| Slide | Description | Method |
|-------|-------------|--------|
| 3 | Timeline of NLP eras | TikZ timeline |
| 5 | Bag-of-Words matrix | LaTeX tabular |
| 6 | Dog park analogy | TikZ illustration |
| 7 | TF-IDF formula + table | LaTeX math + tabular |
| 8 | TF-IDF worked example | TikZ |
| 11 | Matrix factorization A ≈ W × H | TikZ |
| 13 | LDA graphical model | TikZ |
| 14 | BERTopic pipeline | TikZ flow diagram |
| 16 | Vector space scatter plot | TikZ |
| 17 | King-Queen vector arithmetic | TikZ |
| 18 | Sliding window training | TikZ |
| 19 | UFO in village (vector distance) | TikZ |
| 20 | t-SNE cluster visualization | TikZ |
| 24 | Assembly line transformer layers | TikZ |
| 25 | Encoder/Decoder comparison table | LaTeX tabular |
| 26 | Similarity comparison table | LaTeX tabular |
| 27 | LLM landscape map | TikZ |
| 29 | Cost bar chart | pgfplots |
| 31 | DeepSeek impact timeline | TikZ |
| 32 | Context window bar chart | pgfplots |
| 34 | Benchmark comparison | LaTeX tabular |
| 36 | HLE accuracy bar chart | pgfplots |
| 38 | Innovation space scatter | TikZ |
| 41 | Task exposure pie chart | TikZ |
| 42 | Three-day plan overview | TikZ |
| 43 | Professional's playbook escalation | TikZ |

## Generated Images (in gemini/)

| File | Description | Used on |
|------|-------------|---------|
| gemini/bertopic.png | BERTopic pipeline flow diagram | Slide 14 (optional enhancement) |
| gemini/title.png | Title slide illustration | Slide 1 (optional enhancement) |

## JSON Prompts for LLM Image Generation (technical charts)

These are kept for potential future generation via nano-banana/Gemini:

| File | Description | Notes |
|------|-------------|-------|
| 06_attention_mechanism.json | Self-attention visualization | Could enhance slide 23 |
| 07_cost_collapse_chart.json | Cost per token comparison | pgfplots version exists |
| 08_hle_benchmark.json | HLE accuracy chart | pgfplots version exists |
| 09_context_windows.json | Context window growth | pgfplots version exists |
| 10_vector_space.json | Embedding space visualization | TikZ version exists |
| 11_topic_model_visualization.json | Full-slide topic-model flow graphic | Inserted as standalone frame in `slides_round1.tex` |

## Source Citations
- BERTopic diagram: Grootendorst, M. (2022). Licensed under MIT.
- Jay Alammar illustrations: jalammar.github.io. Licensed under CC BY-NC-SA 4.0.
