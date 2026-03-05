from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: List[Dict[str, Any]]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    emb = _model.encode(texts, show_progress_bar=False)
    return np.asarray(emb, dtype="float32")