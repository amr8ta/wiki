import streamlit as st
from indexer import build_index, get_embedding
import numpy as np

st.set_page_config(page_title="Confluence Semantic Search", layout="wide")
st.title("🔍 Confluence Semantic Search (BERT + FAISS)")

@st.cache_resource(show_spinner="Building index...")
def load_data():
    return build_index()

index, chunks, metadata = load_data()

query = st.text_input("Enter your search query:")

if query:
    query_vec = get_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, 5)

    st.write(f"### Top {len(indices[0])} results:")

    for i in indices[0]:
        st.markdown(f"#### [{metadata[i]['title']}]({metadata[i]['url']})")
        st.write(chunks[i])
        st.markdown("---")
