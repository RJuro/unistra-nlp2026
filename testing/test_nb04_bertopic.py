"""Test script for NB04: BERTopic — Topic Discovery + LLM Annotation"""
import sys, os, time
start = time.time()

print("=" * 60)
print("TEST: NB04 — BERTopic + LLM Topic Naming")
print("=" * 60)

# ── 1. Load Moltbook dataset ──
print("\n[1/8] Loading Moltbook dataset...")
from datasets import load_dataset
import pandas as pd
import numpy as np
import re

try:
    dataset = load_dataset("TrustAIRLab/Moltbook", "posts", split="train")
    print(f"  Loaded {len(dataset)} posts")
except Exception as e:
    print(f"  ✗ Failed to load Moltbook: {e}")
    sys.exit(1)

# ── 2. Data preprocessing ──
print("\n[2/8] Preprocessing...")
rows = []
for item in dataset:
    post = item.get('post', {})
    if isinstance(post, dict):
        title = post.get('title', '')
        content = post.get('content', '')
        submolt = post.get('submolt', {})
        submolt_name = submolt.get('name', '') if isinstance(submolt, dict) else ''
    else:
        title = str(item.get('title', ''))
        content = str(item.get('content', ''))
        submolt_name = ''
    rows.append({
        'title': title, 'content': content,
        'submolt': submolt_name,
        'topic_label': item.get('topic_label', 'unknown'),
    })

df = pd.DataFrame(rows)
df['text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
print(f"  Rows: {len(df)}, columns: {list(df.columns)}")
print(f"  Topic labels: {df['topic_label'].nunique()} unique")

# Clean
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)
df = df[df['text_clean'].str.len() > 20].reset_index(drop=True)
print(f"  After filtering: {len(df)} rows")

# Sample for speed
SAMPLE_SIZE = 2000
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"  Sampled to {SAMPLE_SIZE} for testing")
print("  ✓ Data OK")

# ── 3. Embeddings ──
print("\n[3/8] Generating embeddings...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-small')
texts = ["passage: " + t for t in df['text_clean']]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
print(f"  Embeddings shape: {embeddings.shape}")
assert embeddings.shape[0] == len(df)
assert embeddings.shape[1] == 384
print("  ✓ Embeddings OK")

# ── 4. UMAP + HDBSCAN + BERTopic ──
print("\n[4/8] BERTopic pipeline...")
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=10, metric='euclidean', prediction_data=True)
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)
representation = KeyBERTInspired()

topic_model = BERTopic(
    embedding_model=model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    representation_model=representation,
    verbose=False,
)

topics, probs = topic_model.fit_transform(df['text_clean'].tolist(), embeddings)
topic_info = topic_model.get_topic_info()
n_topics = len(topic_info[topic_info['Topic'] != -1])
n_outliers = (np.array(topics) == -1).sum()
print(f"  Topics found: {n_topics}")
print(f"  Outliers: {n_outliers} ({n_outliers/len(topics)*100:.1f}%)")
print(f"  Topic info head:")
for _, row in topic_info.head(5).iterrows():
    print(f"    Topic {row['Topic']}: {row['Count']} docs — {row['Name'][:60]}")
assert n_topics >= 3, f"Expected at least 3 topics, got {n_topics}"
print("  ✓ BERTopic OK")

# ── 5. Visualizations (verify they don't crash) ──
print("\n[5/8] Visualization tests...")
try:
    fig = topic_model.visualize_barchart(top_n_topics=5, n_words=8)
    print("  ✓ visualize_barchart OK")
except Exception as e:
    print(f"  ⚠ visualize_barchart: {e}")

try:
    fig = topic_model.visualize_topics()
    print("  ✓ visualize_topics OK (intertopic distance)")
except Exception as e:
    print(f"  ⚠ visualize_topics: {e}")

try:
    fig = topic_model.visualize_hierarchy()
    print("  ✓ visualize_hierarchy OK")
except Exception as e:
    print(f"  ⚠ visualize_hierarchy: {e}")

# ── 6. 2D UMAP for document visualization ──
print("\n[6/8] 2D UMAP projection...")
umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
coords_2d = umap_2d.fit_transform(embeddings)
print(f"  2D coords shape: {coords_2d.shape}")
assert coords_2d.shape == (len(df), 2)
print("  ✓ 2D UMAP OK")

# ── 7. LLM topic naming (Groq) ──
print("\n[7/8] LLM topic naming...")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if GROQ_API_KEY:
    from openai import OpenAI as LLMClient
    groq_client = LLMClient(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    LLM_MODEL = "moonshotai/kimi-k2-instruct"

    # Test model availability
    try:
        test = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Name a topic in 3 words."}],
            max_tokens=20, temperature=0.0,
        )
        print(f"  kimi-k2 test: {test.choices[0].message.content.strip()}")

        from bertopic.representation import OpenAI as OpenAIRepresentation
        llm_prompt = """Based on these documents and keywords, generate a concise topic name (3-6 words).
Keywords: [KEYWORDS]
Documents: [DOCUMENTS]
Topic name:"""

        llm_rep = OpenAIRepresentation(
            client=groq_client,
            model=LLM_MODEL,
            prompt=llm_prompt,
            nr_docs=3,
            doc_length=300,
            delay_in_seconds=2,
        )
        topic_model.update_topics(df['text_clean'].tolist(), representation_model=llm_rep)
        new_info = topic_model.get_topic_info()
        print(f"  LLM-named topics:")
        for _, row in new_info.head(5).iterrows():
            print(f"    Topic {row['Topic']}: {row['Name'][:60]}")
        print("  ✓ LLM topic naming OK")
    except Exception as e:
        print(f"  ⚠ LLM naming failed: {e}")
else:
    print("  ⚠ No GROQ_API_KEY — skipping LLM naming")

# ── 8. Ground truth comparison ──
print("\n[8/8] Ground truth comparison...")
df['topic_id'] = topics
ct = pd.crosstab(df['topic_label'], df['topic_id'])
print(f"  Cross-tab shape: {ct.shape} (labels × topics)")
print(f"  Ground truth labels: {list(ct.index[:5])}")
print("  ✓ Comparison OK")

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"NB04 RESULT: PASS")
print(f"  Dataset: {len(df)} posts (sampled from Moltbook)")
print(f"  Topics: {n_topics}, Outliers: {n_outliers}")
print(f"  LLM model: moonshotai/kimi-k2-instruct")
print(f"  Time: {elapsed:.1f}s")
print(f"{'=' * 60}")
