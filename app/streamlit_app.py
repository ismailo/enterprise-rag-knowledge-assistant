import sys
import os
import hashlib
from pathlib import Path
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_loader import load_documents
from src.chunker import chunk_text
from src.embeddings import embed_chunks
from src.vector_store import create_index
from src.query_embedding import embed_query
from src.hybrid_retriever import build_bm25, hybrid_retrieve
from src.rag_pipeline import generate_answer
from src.eval_grounding import grounding_check


st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("Enterprise RAG Knowledge Assistant")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop" not in st.session_state:
    st.session_state.stop = False
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


def _hash_files(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode("utf-8"))
        h.update(str(len(f.getbuffer())).encode("utf-8"))
    return h.hexdigest()[:16]


def _save_uploads(files) -> None:
    # Clear old uploads
    for old in UPLOAD_DIR.glob("*"):
        old.unlink()

    for f in files:
        out = UPLOAD_DIR / f.name
        out.write_bytes(f.getbuffer())


@st.cache_resource(show_spinner=False)
def build_indexes(cache_key: str):
    """
    Build chunks, FAISS index, BM25 index.
    cache_key changes when uploads change.
    """
    docs = load_documents(str(UPLOAD_DIR))
    chunks = []

    for doc in docs:
        for page in doc["pages"]:
            page_chunks = chunk_text(page["text"])
            for cid, ch in enumerate(page_chunks):
                chunks.append({
                    "text": ch,
                    "source": doc["source"],
                    "page_num": page["page_num"],
                    "chunk_id": cid
                })

    embeddings = embed_chunks(chunks)
    faiss_index = create_index(embeddings)
    bm25 = build_bm25(chunks)

    return faiss_index, bm25, chunks


# --- Sidebar: uploader + controls
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload PDF/TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.stop = False
            st.rerun()

    with colB:
        if st.button("Stop"):
            st.session_state.stop = True
            st.warning("Stop requested. Submit a new question to continue.")

    st.divider()
    st.caption("Tip: Upload 2–5 small docs for testing.")

# If user uploaded docs, save and build indexes
if uploaded:
    cache_key = _hash_files(uploaded)
    _save_uploads(uploaded)
    with st.spinner("Indexing documents (first time may take a minute)..."):
        faiss_index, bm25, chunks = build_indexes(cache_key)
    st.session_state.index_ready = True
else:
    # fallback: if uploads folder already has files
    existing = list(UPLOAD_DIR.glob("*.pdf")) + list(UPLOAD_DIR.glob("*.txt"))
    if existing:
        cache_key = "existing_uploads"
        faiss_index, bm25, chunks = build_indexes(cache_key)
        st.session_state.index_ready = True

if not st.session_state.index_ready:
    st.info("Upload PDF/TXT documents in the sidebar to start.")
    st.stop()

# --- Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg.get("sources"):
            st.subheader("Sources")
            for s in msg["sources"]:
                with st.expander(f"{s['source']} | p.{s['page_num']} | chunk {s['chunk_id']}"):
                    st.write(s["text"])

        if msg.get("grounding"):
            g = msg["grounding"]
            st.subheader("Grounding check")
            st.write(f"Supported: **{g.get('supported')}**")
            if g.get("unsupported_claims"):
                st.write("Unsupported claims:")
                for c in g["unsupported_claims"]:
                    st.write(f"- {c}")
            if g.get("notes"):
                st.caption(g["notes"])

# --- Input (ChatGPT-like)
prompt = st.chat_input("Ask a question about your documents")

if prompt:
    st.session_state.stop = False  # reset stop for new question
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            q_emb = embed_query(prompt)

            # Hybrid retrieval (semantic + keyword)
            top_chunks = hybrid_retrieve(
                query=prompt,
                query_emb=q_emb,
                faiss_index=faiss_index,
                bm25=bm25,
                chunks=chunks,
                k_final=5,
                k_semantic=10,
                k_keyword=10
            )

            if st.session_state.stop:
                st.stop()

            answer = generate_answer(prompt, top_chunks)
            st.write(answer)

            # Grounding verification
            g = grounding_check(prompt, answer, top_chunks)

            # show sources
            st.subheader("Sources")
            for s in top_chunks:
                with st.expander(f"{s['source']} | p.{s['page_num']} | chunk {s['chunk_id']}"):
                    st.write(s["text"])

            st.subheader("Grounding check")
            st.write(f"Supported: **{g.get('supported')}**")
            if g.get("unsupported_claims"):
                st.write("Unsupported claims:")
                for c in g["unsupported_claims"]:
                    st.write(f"- {c}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": top_chunks,
        "grounding": g
    })