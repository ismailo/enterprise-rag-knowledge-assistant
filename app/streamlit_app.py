import sys
import os
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_loader import load_documents
from src.chunker import chunk_text
from src.embeddings import embed_chunks
from src.vector_store import create_index
from src.query_embedding import embed_query
from src.retriever import retrieve
from src.rag_pipeline import generate_answer


st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")

st.title("Enterprise RAG Knowledge Assistant")

# session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False


# Load and index documents once
@st.cache_resource
def load_index():

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

    return index, chunks


index, chunks = load_index()


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if "sources" in msg:
            for s in msg["sources"]:
                with st.expander(f"{s['source']} | Chunk {s['chunk_id']}"):
                    st.write(s["text"])


# Input box
prompt = st.chat_input("Ask a question about your documents")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            query_vector = embed_query(prompt)

            context_chunks = retrieve(query_vector, index, chunks)

            if st.session_state.stop_generation:
                st.stop()

            answer = generate_answer(prompt, context_chunks)

            st.write(answer)

            st.subheader("Sources")

            for chunk in context_chunks:
                with st.expander(f"{chunk['source']} | Chunk {chunk['chunk_id']}"):
                    st.write(chunk["text"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": context_chunks
        })


# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("Stop Generation"):
        st.session_state.stop_generation = True