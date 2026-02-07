# Lecturer Briefing (Slides 1–13)

## Scope
Companion talk track for `lecture/slides_round1.tex` (slides 1 to 13).  
Target style: short spoken sentences, clear definitions, explicit transitions.

---

## Slide 1 - Title
Slide goal: Set expectations and establish the practical focus of the lecture.

Narrative:
Welcome to Applied NLP.  
This course is about methods you can actually use.  
We will connect theory to workflows and tools.  
The pacing is progressive: foundations first, then modern systems.  
By the end, you should know what to use, when, and why.

Key term definitions:
- NLP: computational methods for understanding and generating language.
- Applied: focused on solving concrete tasks, not only theory.

Talk track transition: To start, I will show the two tracks that run through the whole lecture.

Likely student question: Is this course mostly conceptual or technical? Suggested answer: It is both; we build intuition first and then operationalize it.

---

## Slide 2 - Two Tracks
Slide goal: Introduce the dual structure: conceptual understanding and text operations.

Narrative:
We run two tracks in parallel.  
First, concepts and intuition: how representation and retrieval work.  
Second, operations on text: classify, extract, retrieve, reason.  
This framing avoids tool tunnel vision.  
Methods change, but core operations stay stable.  
If you master operations, you can adapt to new models quickly.

Key term definitions:
- Representation: how text is encoded for computation.
- Retrieval: finding relevant evidence in a large corpus.
- Reason: synthesizing evidence into a usable output.

Talk track transition: Next, we place these ideas on a timeline of method evolution.

Likely student question: Why discuss operations so early? Suggested answer: Because they remain stable across model generations and guide system design.

---

## Slide 3 - Roadmap
Slide goal: Show historical progression and what each era unlocked.

Narrative:
This timeline is not just dates.  
Each era adds a new capability layer.  
TF-IDF gave scalable lexical matching.  
Topic models added latent structure.  
Embeddings and transformers improved semantic context.  
LLMs expanded general-purpose text operations.  
In practice, we combine methods instead of replacing everything.

Key term definitions:
- Latent structure: hidden thematic patterns inferred from word usage.
- Contextualization: representing words differently depending on surrounding text.

Talk track transition: To understand modern systems, we begin with the oldest durable idea: word counting.

Likely student question: Do old methods still matter after LLMs? Suggested answer: Yes; many production systems still rely on lexical and hybrid retrieval.

---

## Slide 4 - Word Counts and Information Theory
Slide goal: Connect simple counts to an information-theoretic justification for IDF.

Narrative:
Raw counts are a useful starting point.  
But frequent words dominate and reduce discrimination.  
Information theory gives a cleaner principle.  
Rare events carry higher surprisal.  
IDF is the retrieval version of that idea.  
So rare but relevant terms deserve larger weight.

Key term definitions:
- Self-information (surprisal): how unexpected an event is, measured as `-log p(x)`.
- IDF: inverse document frequency; higher when a term appears in fewer documents.

Talk track transition: Before TF-IDF, let us formalize the baseline representation: bag of words.

Likely student question: Is IDF always better than raw frequency? Suggested answer: Usually for retrieval and discrimination, but domain and task still matter.

---

## Slide 5 - Bag of Words
Slide goal: Explain the vector representation that underlies early NLP pipelines.

Narrative:
Bag of words turns each document into a frequency vector.  
This is simple and computationally cheap.  
It ignores word order and syntax.  
That is a limitation, but it can still work well.  
For classification and indexing, this baseline is often strong.  
Many modern methods still start from similar matrix structures.

Key term definitions:
- Bag of words: representation using token counts without order.
- Feature vector: numeric encoding of a document for modeling.

Talk track transition: Now we improve bag of words by separating generic terms from distinctive signal.

Likely student question: If order is ignored, why can results still be good? Suggested answer: Many tasks are driven by term presence and frequency more than full syntax.

---

## Slide 6 - TF-IDF Intuition
Slide goal: Build intuition for local frequency times global rarity.

Narrative:
Imagine a stream of repetitive announcements.  
Some terms appear everywhere and carry little signal.  
A rare domain-specific phrase stands out immediately.  
TF captures local emphasis inside one document.  
IDF discounts globally common words.  
The product highlights what is both present and distinctive.

Key term definitions:
- TF: term frequency inside one document.
- Global rarity: how uncommon a term is across the corpus.

Talk track transition: With the intuition set, we can now read the formula directly.

Likely student question: Can a term with low TF still matter? Suggested answer: Yes; if it is very rare globally, IDF can make it highly informative.

---

## Slide 7 - TF-IDF Math
Slide goal: Translate the intuition into the standard weighting equation.

Narrative:
The formula has two factors.  
First factor: local frequency in a document.  
Second factor: inverse document frequency in the corpus.  
The table shows how common terms get small weights.  
Rare terms can dominate despite lower raw counts.  
This is the core mechanism behind lexical retrieval.

Key term definitions:
- Document frequency (DF): number of documents containing a term.
- Weighting: converting raw counts into discriminative scores.

Talk track transition: Let us run one full numeric example to make the scale effects concrete.

Likely student question: Why use logarithms in IDF? Suggested answer: Log compresses extreme ratios and stabilizes weighting.

---

## Slide 8 - TF-IDF Worked Example
Slide goal: Provide concrete numeric intuition with interpretable magnitudes.

Narrative:
We use one synthetic announcement with 80 tokens.  
Then we compare common and rare terms.  
Common terms keep low IDF and low final weight.  
Rare domain terms gain much larger scores.  
This slide is meant for calibration, not memorization.  
The key pattern is relative contrast, not exact decimals.

Key term definitions:
- Discriminative term: term that helps distinguish one document from others.
- Weight contrast: difference in scores between common and rare terms.

Talk track transition: TF-IDF is foundational, but modern search usually uses an improved descendant.

Likely student question: Should we hand-compute TF-IDF in practice? Suggested answer: Usually no; libraries do it, but understanding the mechanics is essential.

---

## Slide 9 - BM25
Slide goal: Show why BM25 became the dominant lexical ranking function.

Narrative:
BM25 extends TF-IDF with practical ranking corrections.  
The key addition is term-frequency saturation.  
More repeats help, but with diminishing returns.  
This avoids over-rewarding long documents.  
That is why BM25 remains default in major search engines.  
Recent variants like BMX push this family further, not away from it.

Key term definitions:
- Saturation: nonlinear gain where extra occurrences contribute less.
- Lexical retrieval: ranking based on token overlap and term weighting.

Talk track transition: After lexical ranking, we move to latent thematic structure with topic models.

Likely student question: If BM25 is old, why is it still used? Suggested answer: It is robust, fast, interpretable, and often hard to beat without strong hybrid setups.

---

## Slide 10 - Uncovering Hidden Relationships
Slide goal: Contrast LSA, LDA, and NMF as three routes to latent structure.

Narrative:
All three methods compress document-word data into fewer dimensions.  
LSA uses linear algebra to capture co-occurrence patterns.  
LDA uses a probabilistic generative story for topics.  
NMF enforces non-negativity and often improves interpretability.  
Method choice depends on research goal, corpus type, and transparency needs.  
Evaluation is partly quantitative and partly human judgment.

Key term definitions:
- Topic model: method that infers recurring semantic themes.
- Interpretability: how easily humans can explain model outputs.

Talk track transition: To ground this, we now visualize factorization as a document-topic-word map.

Likely student question: Which method should we use first? Suggested answer: Start with the method that best matches your interpretability and workflow constraints, then compare outputs.

---

## Slide 11 - Matrix Factorization
Slide goal: Make the factorization view intuitive before returning to formal probabilistic modeling.

Narrative:
Think of factorization as a two-step map.  
Documents map to topic mixtures.  
Topics map to weighted word profiles.  
`W` tells us topic proportions per document.  
`H` tells us which words define each topic.  
This decomposition is the shared geometry behind several topic methods.

Key term definitions:
- Factorization: approximating one matrix as a product of smaller matrices.
- Topic profile: weighted set of words characterizing a theme.

Talk track transition: With the map in mind, we can interpret concrete topics from a large corpus.

Likely student question: Are factors always meaningful topics? Suggested answer: Not always; semantic quality depends on preprocessing, data quality, and model settings.

---

## Slide 12 - Discovering Topics in Text
Slide goal: Demonstrate mixed-topic documents and practical interpretation.

Narrative:
This example shows four coherent themes from a larger corpus.  
Real documents are usually mixtures, not single-topic units.  
That mixture view is critical for social-science interpretation.  
An article can be partly technology and partly health.  
Topic proportions become usable variables for downstream analysis.  
This is where qualitative interpretation meets quantitative modeling.

Key term definitions:
- Topic mixture: percentage-like distribution of themes per document.
- Corpus: the full collection of documents under study.

Talk track transition: Next, we open the probabilistic mechanics that produce these mixtures in LDA.

Likely student question: Can we force one topic per document? Suggested answer: You can, but it often distorts reality and loses useful nuance.

---

## Slide 13 - LDA: The Generative Model
Slide goal: Explain the LDA generative process in both formal and intuitive language.

Narrative:
LDA assumes each document has a hidden topic distribution.  
For each token position, a topic is sampled first.  
Then a word is sampled from that topic's word distribution.  
So documents become mixtures of latent themes.  
Inference works backward from observed words to hidden structure.  
The model is simple enough to explain and rich enough to be useful.

Key term definitions:
- `Dir(alpha)`: prior over document-level topic mixtures.
- `z_{d,n}`: latent topic assignment for token position `n`.
- `phi_k`: word distribution for topic `k`.

Talk track transition: From here, we can compare classical LDA to modern embedding-based topic modeling.

Likely student question: Are topics real entities or modeling artifacts? Suggested answer: They are statistical constructs; usefulness depends on coherence, stability, and domain validation.

---

## References Used in This Briefing
- `shannon1948`
- `sparckjones1972`
- `aizawa2003`
- `deerwester1990`
- `lee1999`
- `blei2003`
- `robertson1994`
- `robertson2009`
- `li2024bmx`
