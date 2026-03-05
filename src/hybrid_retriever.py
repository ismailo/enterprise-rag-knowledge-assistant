from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized)


def _semantic_topk(query_emb: np.ndarray, faiss_index, k: int) -> List[int]:
    distances, indices = faiss_index.search(query_emb, k)
    return [int(i) for i in indices[0] if int(i) >= 0]


def _bm25_topk(query: str, bm25: BM25Okapi, k: int) -> List[int]:
    scores = bm25.get_scores(_tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    return [int(i) for i in top_idx]


def hybrid_retrieve(
    query: str,
    query_emb: np.ndarray,
    faiss_index,
    bm25: BM25Okapi,
    chunks: List[Dict[str, Any]],
    k_final: int = 5,
    k_semantic: int = 8,
    k_keyword: int = 8,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval = semantic + keyword with simple rank-fusion.
    Returns top k_final chunk dicts.
    """

    sem = _semantic_topk(query_emb, faiss_index, k_semantic)
    kw = _bm25_topk(query, bm25, k_keyword)

    # Rank fusion: give points based on rank position
    scores = {}
    for rank, idx in enumerate(sem):
        scores[idx] = scores.get(idx, 0.0) + (1.0 / (rank + 1))
    for rank, idx in enumerate(kw):
        scores[idx] = scores.get(idx, 0.0) + (1.0 / (rank + 1))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    chosen = [idx for idx, _ in ranked[:k_final]]

    return [chunks[i] for i in chosen]