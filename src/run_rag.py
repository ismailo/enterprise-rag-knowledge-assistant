from document_loader import load_documents
from chunker import chunk_text
from embeddings import embed_chunks
from vector_store import create_index
from query_embedding import embed_query
from retriever import retrieve
from rag_pipeline import generate_answer

docs = load_documents("data/sample_docs")

chunks = []

for doc in docs:
    text_chunks = chunk_text(doc["text"])

    for i, chunk in enumerate(text_chunks):
        chunks.append({
            "text": chunk,
            "source": doc["source"],
            "chunk_id": i
        })

embeddings = embed_chunks(chunks)

index = create_index(embeddings)

while True:

    query = input("\nAsk a question: ")

    if query == "exit":
        break

    query_vector = embed_query(query)
    context_chunks = retrieve(query_vector, index, chunks)
    answer = generate_answer(query, context_chunks)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for chunk in context_chunks:
        print(f"Chunk {chunk['chunk_id']}")