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
    chunks.extend(chunk_text(doc["text"]))

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