from sentence_transformers import CrossEncoder


# Cross-encoder model trained for ranking
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query, chunks, top_k=5):
    """
    Rerank retrieved chunks using cross encoder scoring.
    """

    pairs = []

    for chunk in chunks:
        pairs.append([query, chunk["text"]])

    scores = model.predict(pairs)

    ranked = []

    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        ranked.append(chunk_copy)

    ranked = sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)

    return ranked[:top_k]