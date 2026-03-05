import faiss
import numpy as np


def create_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    embeddings = np.asarray(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index