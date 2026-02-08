#!/usr/bin/env python3
"""Test NB07: Cross-encoder Reranking"""

import sys
import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset

def load_scifact():
    corpus = load_dataset("mteb/scifact", "corpus", split="corpus").to_pandas()
    queries = load_dataset("mteb/scifact", "queries", split="queries").to_pandas()
    qrels_train = load_dataset("mteb/scifact", "default", split="train").to_pandas()
    qrels_test = load_dataset("mteb/scifact", "default", split="test").to_pandas()

    for df in [corpus, queries]:
        df["_id"] = df["_id"].astype(str)
    for df in [qrels_train, qrels_test]:
        df["query-id"] = df["query-id"].astype(str)
        df["corpus-id"] = df["corpus-id"].astype(str)

    corpus["title"] = corpus["title"].fillna("").astype(str)
    corpus["text"] = corpus["text"].fillna("").astype(str)
    corpus["full_text"] = (corpus["title"].str.strip() + ". " + corpus["text"].str.strip()).str.strip(" .")
    corpus_df = corpus.rename(columns={"_id": "doc_id"})[["doc_id", "title", "text", "full_text"]]
    queries_df = queries.rename(columns={"_id": "query_id", "text": "query"})[["query_id", "query"]]

    qrels = pd.concat([qrels_train, qrels_test], ignore_index=True)
    relevant_docs = qrels.groupby("query-id")["corpus-id"].apply(set).to_dict()

    return corpus_df.reset_index(drop=True), queries_df, relevant_docs


def main():
    print("=" * 60)
    print("TEST NB07: Cross-encoder Reranking")
    print("=" * 60)

    # --- 1. Load data + bi-encoder ---
    print("\n[1] Loading SciFact + encoding...")
    corpus_df, queries_df, relevant_docs = load_scifact()
    print(f"  Corpus: {len(corpus_df)}, Queries: {len(queries_df)}, Queries with qrels: {len(relevant_docs)}")

    bi_encoder = SentenceTransformer("intfloat/e5-small")
    corpus_inputs = [f"passage: {t.strip()}" for t in corpus_df["full_text"].tolist()]

    start = time.time()
    corpus_embeddings = bi_encoder.encode(corpus_inputs, show_progress_bar=True, batch_size=64, normalize_embeddings=True).astype("float32")
    print(f"  Encoded in {time.time()-start:.1f}s")

    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)

    # --- 2. Load cross-encoder ---
    print("\n[2] Loading cross-encoder...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    print("  Cross-encoder loaded!")

    # --- 3. Retrieve and rerank ---
    print("\n[3] Testing retrieve_and_rerank...")

    def retrieve_and_rerank(query, top_k_retrieve=20, top_k_final=5):
        q_emb = bi_encoder.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
        bi_scores, bi_indices = index.search(q_emb, top_k_retrieve)

        pairs = [(query, corpus_df.iloc[idx]["full_text"]) for idx in bi_indices[0]]
        cross_scores = cross_encoder.predict(pairs)

        reranked = sorted(
            zip(bi_indices[0], bi_scores[0], cross_scores),
            key=lambda x: x[2], reverse=True
        )

        results = []
        for rank, (idx, bi_score, ce_score) in enumerate(reranked[:top_k_final], 1):
            results.append({
                "rank": rank,
                "doc_id": corpus_df.iloc[idx]["doc_id"],
                "bi_score": float(bi_score),
                "ce_score": float(ce_score),
                "title": corpus_df.iloc[idx]["title"],
                "text": corpus_df.iloc[idx]["full_text"][:200],
            })
        return pd.DataFrame(results)

    def bi_encoder_only(query, top_k=5):
        q_emb = bi_encoder.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(q_emb, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            results.append({
                "rank": rank,
                "doc_id": corpus_df.iloc[idx]["doc_id"],
                "score": float(score),
                "title": corpus_df.iloc[idx]["title"],
            })
        return pd.DataFrame(results)

    query = "What are the risk factors for developing lung cancer?"
    bi_results = bi_encoder_only(query)
    reranked_results = retrieve_and_rerank(query)
    print(f"\n  Query: {query}")
    print("  Bi-encoder only:")
    for _, r in bi_results.iterrows():
        print(f"    [{r['rank']}] ({r['score']:.3f}) {r['title'][:60]}")
    print("  Reranked:")
    for _, r in reranked_results.iterrows():
        print(f"    [{r['rank']}] (bi:{r['bi_score']:.3f} -> ce:{r['ce_score']:.3f}) {r['title'][:60]}")
    print("  PASS: Retrieve and rerank works")

    # --- 4. Ground-truth evaluation ---
    print("\n[4] Ground-truth evaluation (NDCG@k, Precision@k)...")

    def precision_at_k(retrieved_ids, relevant_set, k):
        hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
        return hits / k

    def ndcg_at_k(retrieved_ids, relevant_set, k):
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0

    bi_p5, bi_ndcg10 = [], []
    re_p5, re_ndcg10 = [], []

    eval_count = 0
    for _, row in queries_df.iterrows():
        qid = row["query_id"]
        if qid not in relevant_docs:
            continue

        query_text = row["query"]
        gold = relevant_docs[qid]

        bi_res = bi_encoder_only(query_text, top_k=20)
        bi_ids = bi_res["doc_id"].tolist()

        re_res = retrieve_and_rerank(query_text, top_k_retrieve=50, top_k_final=20)
        re_ids = re_res["doc_id"].tolist()

        bi_p5.append(precision_at_k(bi_ids, gold, 5))
        bi_ndcg10.append(ndcg_at_k(bi_ids, gold, 10))
        re_p5.append(precision_at_k(re_ids, gold, 5))
        re_ndcg10.append(ndcg_at_k(re_ids, gold, 10))
        eval_count += 1

    print(f"  Evaluated on {eval_count} queries")
    print(f"\n  {'Metric':<20} {'Bi-encoder':>12} {'+ Reranking':>12} {'Delta':>10}")
    print("  " + "-" * 56)
    for name, bi_vals, re_vals in [
        ("Precision@5", bi_p5, re_p5),
        ("NDCG@10", bi_ndcg10, re_ndcg10),
    ]:
        bm, rm = np.mean(bi_vals), np.mean(re_vals)
        print(f"    {name:<18} {bm:>10.1%}   {rm:>10.1%}   {rm - bm:>+9.1%}")

    print("  PASS: Ground-truth evaluation completed")

    # --- 5. Speed comparison ---
    print("\n[5] Speed comparison...")
    query = "effects of air pollution on respiratory health"

    start = time.time()
    for _ in range(5):
        bi_encoder_only(query)
    bi_time = (time.time() - start) / 5

    start = time.time()
    for _ in range(5):
        retrieve_and_rerank(query)
    rerank_time = (time.time() - start) / 5

    print(f"  Bi-encoder only:  {bi_time*1000:.1f} ms/query")
    print(f"  With reranking:   {rerank_time*1000:.1f} ms/query")
    print(f"  Overhead: {(rerank_time-bi_time)*1000:.1f} ms ({rerank_time/bi_time:.1f}x)")
    print("  PASS: Speed comparison completed")

    print("\n" + "=" * 60)
    print("NB07: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
