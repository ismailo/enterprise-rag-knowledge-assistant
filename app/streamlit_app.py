import streamlit as st

st.title("Enterprise RAG Knowledge Assistant")

query = st.text_input("Ask a question")

if query:
    st.write("Processing...")