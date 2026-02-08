#!/usr/bin/env python3
"""Test NB06: FAISS Retrieval + Semantic Search"""

import sys
import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

def main():
    print("=" * 60)
    print("TEST NB06: FAISS Retrieval + Semantic Search")
    print("=" * 60)

    # --- 1. Load SciFact ---
    print("\n[1] Loading SciFact dataset...")
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus").to_pandas()
    queries = load_dataset("mteb/scifact", "queries", split="queries").to_pandas()

    corpus["_id"] = corpus["_id"].astype(str)
    queries["_id"] = queries["_id"].astype(str)
    corpus["title"] = corpus["title"].fillna("").astype(str)
    corpus["text"] = corpus["text"].fillna("").astype(str)
    corpus["full_text"] = (corpus["title"].str.strip() + ". " + corpus["text"].str.strip()).str.strip(" .")
    corpus_df = corpus.rename(columns={"_id": "doc_id"})[["doc_id", "title", "text", "full_text"]]
    queries_df = queries.rename(columns={"_id": "query_id", "text": "query"})[["query_id", "query"]]

    print(f"  Corpus: {len(corpus_df)} docs, Queries: {len(queries_df)}")

    # --- 2. Encode with e5-small ---
    print("\n[2] Encoding corpus with intfloat/e5-small...")
    model = SentenceTransformer("intfloat/e5-small")
    corpus_inputs = [f"passage: {t.strip()}" for t in corpus_df["full_text"].tolist()]

    start = time.time()
    corpus_embeddings = model.encode(corpus_inputs, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    elapsed = time.time() - start
    print(f"  Encoded {len(corpus_embeddings)} docs in {elapsed:.1f}s")
    print(f"  Embedding shape: {corpus_embeddings.shape}")

    # --- 3. Build FAISS index ---
    print("\n[3] Building FAISS IndexFlatIP...")
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings.astype("float32"))
    print(f"  Index: {index.ntotal} vectors, {dimension} dims")

    # --- 4. Test queries ---
    print("\n[4] Running test queries...")
    test_queries = [
        "effects of climate change on biodiversity",
        "how do vaccines work",
        "machine learning for medical diagnosis",
        "genetic factors in cancer risk",
        "air pollution and respiratory disease",
    ]

    for q in test_queries:
        q_emb = model.encode([f"query: {q}"], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_emb, 3)
        print(f"\n  Query: {q}")
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            title = corpus_df.iloc[idx]["title"]
            print(f"    [{rank}] (score: {score:.3f}) {title[:60]}")
        assert len(scores[0]) == 3, f"Expected 3 results, got {len(scores[0])}"
        assert scores[0][0] > 0, "Top score should be positive"
    print("  PASS: All queries returned non-empty, meaningful results")

    # --- 5. Keyword-proxy evaluation ---
    print("\n[5] Keyword-proxy Precision@5...")
    eval_queries = [
        {"query": "vaccine effectiveness against viral infections",
         "relevant_terms": ["vaccine", "immunization", "viral", "antibod"]},
        {"query": "genetic mutations and cancer development",
         "relevant_terms": ["genetic", "mutation", "cancer", "tumor", "oncog"]},
        {"query": "impact of air pollution on health",
         "relevant_terms": ["pollution", "air", "respiratory", "particulate"]},
    ]

    for eq in eval_queries:
        q_emb = model.encode([f"query: {eq['query']}"], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_emb, 5)
        relevant = 0
        for idx in indices[0]:
            doc_lower = corpus_df.iloc[idx]["full_text"].lower()
            if any(term in doc_lower for term in eq["relevant_terms"]):
                relevant += 1
        precision = relevant / 5
        print(f"  '{eq['query'][:50]}...' → P@5: {precision:.0%} ({relevant}/5)")
    print("  PASS: Keyword-proxy evaluation completed")

    # --- 6. Multilingual model ---
    print("\n[6] Loading multilingual-e5-small...")
    ml_model = SentenceTransformer("intfloat/multilingual-e5-small")
    ml_inputs = [f"passage: {t.strip()}" for t in corpus_df["full_text"].tolist()]
    ml_embeddings = ml_model.encode(ml_inputs, show_progress_bar=True, batch_size=64, normalize_embeddings=True)

    ml_index = faiss.IndexFlatIP(ml_embeddings.shape[1])
    ml_index.add(ml_embeddings.astype("float32"))
    print(f"  Multilingual index: {ml_index.ntotal} vectors, {ml_embeddings.shape[1]} dims")

    # --- 7. Cross-lingual queries ---
    print("\n[7] Cross-lingual retrieval test...")
    cross_lingual_queries = [
        ("English", "effects of vaccination on immune response"),
        ("French", "effets de la vaccination sur la réponse immunitaire"),
        ("German", "Auswirkungen der Impfung auf die Immunantwort"),
        ("Spanish", "efectos de la vacunación en la respuesta inmune"),
    ]

    top_ids_per_lang = []
    for lang, query in cross_lingual_queries:
        q_emb = ml_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
        scores, indices = ml_index.search(q_emb, 3)
        top_ids = list(indices[0])
        top_ids_per_lang.append(set(top_ids))
        print(f"  [{lang}] top-3 indices: {top_ids}")

    # Check overlap between English and other languages
    en_set = top_ids_per_lang[0]
    for i, (lang, _) in enumerate(cross_lingual_queries[1:], 1):
        overlap = len(en_set & top_ids_per_lang[i])
        print(f"  EN ∩ {lang}: {overlap}/3 shared")
    print("  PASS: Cross-lingual retrieval completed")

    # --- 8. ChromaDB ---
    print("\n[8] ChromaDB test...")
    import chromadb
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="test_scifact",
        metadata={"hnsw:space": "cosine"}
    )

    N = min(500, len(corpus_df))
    docs = corpus_df["full_text"].tolist()[:N]
    titles = corpus_df["title"].tolist()[:N]
    ids = [str(i) for i in range(N)]
    metadatas = [{"title": t} for t in titles]

    BATCH = 100
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        collection.add(ids=ids[start:end], documents=docs[start:end], metadatas=metadatas[start:end])

    results = collection.query(
        query_texts=["effects of vaccination on immune response"],
        n_results=5,
        include=["metadatas", "distances"]
    )
    print("  ChromaDB results:")
    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
        print(f"    [{i}] (dist: {dist:.3f}) {meta['title'][:60]}")
    assert len(results["metadatas"][0]) == 5, "ChromaDB should return 5 results"
    print("  PASS: ChromaDB working")

    print("\n" + "=" * 60)
    print("NB06: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
