from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_query(query: str) -> np.ndarray:
    emb = _model.encode([query], show_progress_bar=False)
    return np.asarray(emb, dtype="float32")