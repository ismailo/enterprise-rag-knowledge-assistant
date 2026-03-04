def retrieve(query_embedding, index, chunks, k=3):
    distances, indices = index.search(query_embedding, k)

    results = [chunks[i] for i in indices[0]]

    return results