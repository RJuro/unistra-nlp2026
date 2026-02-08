"""Test script for NB02: Sentence Embeddings"""
import sys, time
start = time.time()

print("=" * 60)
print("TEST: NB02 — Sentence Embeddings")
print("=" * 60)

# ── 1. Data loading (same as NB01) ──
print("\n[1/8] Loading dataset...")
import pandas as pd
import numpy as np
import re

DATA_URL = "https://raw.githubusercontent.com/RJuro/unistra-nlp2026/main/data/dk_posts_synth_en_processed.json"
df = pd.read_json(DATA_URL)
df['text'] = df['title'] + ' . ' + df['selftext']
df['text_clean'] = df['text'].apply(lambda t: re.sub(r'\s+', ' ', t.lower()).strip())
print(f"  Loaded {len(df)} rows")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print("  ✓ Data OK")

# ── 2. Load SBERT model ──
print("\n[2/8] Loading all-MiniLM-L6-v2...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"  Model loaded, embedding dim: {model.get_sentence_embedding_dimension()}")
assert model.get_sentence_embedding_dimension() == 384
print("  ✓ Model OK")

# ── 3. Encode texts ──
print("\n[3/8] Encoding train + test...")
X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)
print(f"  Train embeddings: {X_train_emb.shape}")
print(f"  Test embeddings: {X_test_emb.shape}")
assert X_train_emb.shape == (342, 384), f"Expected (342, 384), got {X_train_emb.shape}"
assert X_test_emb.shape == (115, 384), f"Expected (115, 384), got {X_test_emb.shape}"
print("  ✓ Encoding OK")

# ── 4. SBERT + LR ──
print("\n[4/8] SBERT + LogisticRegression...")
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train_emb, y_train)
y_pred_lr = clf_lr.predict(X_test_emb)
acc_sbert_lr = accuracy_score(y_test, y_pred_lr)
print(f"  Accuracy: {acc_sbert_lr:.4f}")
assert 0.75 < acc_sbert_lr < 0.95, f"SBERT+LR accuracy {acc_sbert_lr} outside range"
print("  ✓ SBERT + LR OK")

# ── 5. SBERT + SVC ──
print("\n[5/8] SBERT + LinearSVC...")
clf_svc = LinearSVC(max_iter=2000)
clf_svc.fit(X_train_emb, y_train)
y_pred_svc = clf_svc.predict(X_test_emb)
acc_sbert_svc = accuracy_score(y_test, y_pred_svc)
print(f"  Accuracy: {acc_sbert_svc:.4f}")
assert 0.75 < acc_sbert_svc < 0.95, f"SBERT+SVC accuracy {acc_sbert_svc} outside range"
print("  ✓ SBERT + SVC OK")

# ── 6. TF-IDF baselines for comparison ──
print("\n[6/8] TF-IDF baselines (for comparison chart)...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

pipe_tfidf_lr = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10_000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
pipe_tfidf_lr.fit(X_train, y_train)
acc_tfidf_lr = accuracy_score(y_test, pipe_tfidf_lr.predict(X_test))

pipe_tfidf_svc = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10_000)),
    ("clf", LinearSVC(max_iter=2000)),
])
pipe_tfidf_svc.fit(X_train, y_train)
acc_tfidf_svc = accuracy_score(y_test, pipe_tfidf_svc.predict(X_test))
print(f"  TF-IDF + LR:  {acc_tfidf_lr:.4f}")
print(f"  TF-IDF + SVC: {acc_tfidf_svc:.4f}")

# Check SBERT beats TF-IDF
print(f"\n  Comparison:")
print(f"    SBERT+SVC ({acc_sbert_svc:.4f}) vs TF-IDF+SVC ({acc_tfidf_svc:.4f})")
if acc_sbert_svc > acc_tfidf_svc:
    print("    ✓ SBERT beats TF-IDF (as expected)")
else:
    print("    ⚠ SBERT did NOT beat TF-IDF — unusual for this dataset")

# ── 7. Label efficiency (quick version — 2 sizes only) ──
print("\n[7/8] Label efficiency experiment (quick)...")
from sklearn.model_selection import StratifiedShuffleSplit

sizes = [25, 100]
for n in sizes:
    if n > len(X_train):
        continue
    # Sample n from training
    idx = np.arange(len(X_train))
    np.random.seed(42)
    # Stratified subsample
    sub_idx = []
    for label in y_train.unique():
        label_idx = np.where(y_train.values == label)[0]
        per_class = max(1, n // y_train.nunique())
        chosen = np.random.choice(label_idx, size=min(per_class, len(label_idx)), replace=False)
        sub_idx.extend(chosen)
    sub_idx = np.array(sub_idx)

    clf_sub = LogisticRegression(max_iter=1000, random_state=42)
    clf_sub.fit(X_train_emb[sub_idx], y_train.values[sub_idx])
    acc_sub = accuracy_score(y_test, clf_sub.predict(X_test_emb))
    print(f"  n={n:3d} → SBERT+LR accuracy: {acc_sub:.4f}")

# ── 8. t-SNE visualization (just verify it runs) ──
print("\n[8/8] t-SNE projection...")
from sklearn.manifold import TSNE
all_emb = np.vstack([X_train_emb, X_test_emb])
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
coords = tsne.fit_transform(all_emb)
print(f"  t-SNE output shape: {coords.shape}")
assert coords.shape == (457, 2), f"Expected (457, 2), got {coords.shape}"
print("  ✓ t-SNE OK")

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"NB02 RESULT: PASS")
print(f"  SBERT+LR:   {acc_sbert_lr:.4f}")
print(f"  SBERT+SVC:  {acc_sbert_svc:.4f}")
print(f"  TF-IDF+LR:  {acc_tfidf_lr:.4f}")
print(f"  TF-IDF+SVC: {acc_tfidf_svc:.4f}")
print(f"  Time: {elapsed:.1f}s")
print(f"{'=' * 60}")
