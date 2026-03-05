def retrieve(query_embedding, index, chunks, k=3):

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        results.append(chunks[i])

    return results