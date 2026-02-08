"""Test script for NB01: TF-IDF + Linear Models"""
import sys, time
start = time.time()

print("=" * 60)
print("TEST: NB01 — TF-IDF + Linear Models")
print("=" * 60)

# ── 1. Data loading ──
print("\n[1/7] Loading dataset...")
import pandas as pd
import numpy as np
import re

DATA_URL = "https://raw.githubusercontent.com/RJuro/unistra-nlp2026/main/data/dk_posts_synth_en_processed.json"
df = pd.read_json(DATA_URL)
print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
print(f"  Labels ({df['label'].nunique()}): {dict(df['label'].value_counts())}")
assert len(df) >= 400, f"Expected ~457 rows, got {len(df)}"
assert df['label'].nunique() == 8, f"Expected 8 classes, got {df['label'].nunique()}"
print("  ✓ Dataset OK")

# ── 2. Text preprocessing ──
print("\n[2/7] Text preprocessing...")
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['text'] = df['title'] + ' . ' + df['selftext']
df['text_clean'] = df['text'].apply(clean_text)
assert df['text_clean'].str.len().min() > 0, "Empty texts found after cleaning"
print(f"  Text lengths: min={df['text_clean'].str.len().min()}, max={df['text_clean'].str.len().max()}, mean={df['text_clean'].str.len().mean():.0f}")
print("  ✓ Preprocessing OK")

# ── 3. Train/test split ──
print("\n[3/7] Train/test split...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
assert len(X_train) == 342, f"Expected 342 train, got {len(X_train)}"
assert len(X_test) == 115, f"Expected 115 test, got {len(X_test)}"
# Check stratification
train_dist = y_train.value_counts()
assert train_dist.min() >= 30, "Stratification failed — some classes too small in train"
print("  ✓ Split OK (stratified)")

# ── 4. TF-IDF + Logistic Regression ──
print("\n[4/7] TF-IDF + Logistic Regression...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pipe_lr = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"  Accuracy: {acc_lr:.4f}")
assert 0.70 < acc_lr < 0.95, f"LR accuracy {acc_lr} outside expected range (0.70-0.95)"
print("  ✓ Logistic Regression OK")

# ── 5. TF-IDF + LinearSVC ──
print("\n[5/7] TF-IDF + LinearSVC...")
pipe_svc = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10_000)),
    ("clf", LinearSVC(max_iter=2000)),
])
pipe_svc.fit(X_train, y_train)
y_pred_svc = pipe_svc.predict(X_test)
acc_svc = accuracy_score(y_test, y_pred_svc)
print(f"  Accuracy: {acc_svc:.4f}")
assert 0.70 < acc_svc < 0.95, f"SVC accuracy {acc_svc} outside expected range (0.70-0.95)"
print("  ✓ LinearSVC OK")

# ── 6. Evaluation ──
print("\n[6/7] Evaluation...")
best_name = "Linear SVC" if acc_svc >= acc_lr else "Logistic Regression"
best_pred = y_pred_svc if acc_svc >= acc_lr else y_pred_lr
print(f"  Best model: {best_name}")
print(f"\n  Classification Report ({best_name}):")
print(classification_report(y_test, best_pred, zero_division=0))

cm = confusion_matrix(y_test, best_pred)
print(f"  Confusion matrix shape: {cm.shape}")
assert cm.shape == (8, 8), f"Expected 8x8 confusion matrix, got {cm.shape}"

# ── 7. Top features + error analysis ──
print("\n[7/7] Top features + error analysis...")
# Top features from LR
tfidf = pipe_lr.named_steps['tfidf']
clf = pipe_lr.named_steps['clf']
feature_names = tfidf.get_feature_names_out()
print(f"  Vocabulary size: {len(feature_names)}")
for i, label in enumerate(clf.classes_[:3]):  # Show 3 classes
    top_idx = clf.coef_[i].argsort()[-5:][::-1]
    top_words = [feature_names[j] for j in top_idx]
    print(f"  Top features for '{label}': {top_words}")

# Error analysis
error_mask = y_test != best_pred
n_errors = error_mask.sum()
print(f"\n  Misclassifications: {n_errors}/{len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

# Confidence check (only for LR which has predict_proba)
proba = pipe_lr.predict_proba(X_test)
max_proba = proba.max(axis=1)
wrong_mask = y_test != y_pred_lr
if wrong_mask.any():
    wrong_conf = max_proba[wrong_mask.values]
    print(f"  Avg confidence on wrong predictions: {wrong_conf.mean():.3f}")
    print(f"  Max confidence on wrong prediction: {wrong_conf.max():.3f}")

elapsed = time.time() - start
print(f"\n{'=' * 60}")
print(f"NB01 RESULT: PASS")
print(f"  LR accuracy:  {acc_lr:.4f}")
print(f"  SVC accuracy:  {acc_svc:.4f}")
print(f"  Time: {elapsed:.1f}s")
print(f"{'=' * 60}")
