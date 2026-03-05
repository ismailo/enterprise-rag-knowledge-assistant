from typing import List


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Simple character chunker.
    chunk_size ~900 chars works fine for RAG demos.
    """
    text = text or ""
    text = text.replace("\x00", " ").strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += max(chunk_size - overlap, 1)

    # remove empty
    return [c.strip() for c in chunks if c.strip()]