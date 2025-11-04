import os
from pathlib import Path
import pandas as pd
import streamlit as st


from rag_core import (
    load_artifacts,
    search_hybrid_mmr,
    assemble_context,
    build_prompt,
    ensure_citations,
    cite_str,
    generate_answer_hf,  
)


BASE_DIR      = Path(os.environ.get("RAG_BASE_DIR", "."))
DATA_DIR      = Path(os.environ.get("RAG_DATA_DIR", BASE_DIR / "data"))
FAISS_PATH    = Path(os.environ.get("RAG_FAISS", DATA_DIR / "faiss.index"))
META_PATH     = Path(os.environ.get("RAG_META",  DATA_DIR / "meta.parquet"))
EMB_PATH      = Path(os.environ.get("RAG_EMB",   DATA_DIR / "embeddings_e5.npy"))
BM25_PATH     = Path(os.environ.get("RAG_BM25",  DATA_DIR / "bm25.pkl"))  # optional

DEFAULT_K           = int(os.environ.get("RAG_TOP_K", "5"))
MAX_CTX_TOKENS_DEF  = int(os.environ.get("RAG_MAX_CTX_TOKENS", "1600"))
MAX_NEW_TOKENS_DEF  = int(os.environ.get("RAG_MAX_NEW_TOKENS", "220"))


@st.cache_resource(show_spinner=True)
def _get_artifacts():
    return load_artifacts(
        faiss_path=FAISS_PATH,
        meta_path=META_PATH,
        emb_path=EMB_PATH,
        bm25_path=BM25_PATH if BM25_PATH.exists() else None,
    )

ART = _get_artifacts()


st.set_page_config(page_title="CEADAR RAG", layout="wide")
st.title("🔎 CEADAR RAG — Papers QA")

with st.sidebar:
    st.header("Retrieval")
    k = st.slider("Top-k", min_value=1, max_value=10, value=DEFAULT_K, step=1)
    max_ctx_tokens = st.slider("Context budget (tokens)", 400, 3000, MAX_CTX_TOKENS_DEF, step=100)

    st.header("Generation")
    max_new_tokens = st.slider("Max new tokens", 64, 600, MAX_NEW_TOKENS_DEF, step=16)

    st.caption("Artifacts")
    st.text(f"meta: {META_PATH.name}")
    st.text(f"faiss: {FAISS_PATH.name}")
    st.text(f"emb:  {EMB_PATH.name}")
    st.text(f"bm25: {'yes' if BM25_PATH.exists() else 'no'}")


question = st.text_input("Ask a question about the loaded papers:", value="What is multi-head attention and why is it useful?")
ask = st.button("Run RAG")


if ask and question.strip():
    with st.spinner("Retrieving evidence..."):
        hits = search_hybrid_mmr(question, k=k, artifacts=ART)
        if "citation" not in hits.columns:
            hits["citation"] = hits.apply(cite_str, axis=1)

    with st.expander("Top hits (Hybrid + MMR)", expanded=True):
        cols = ["citation", "heading", "content_type", "page_start", "page_end"]
        st.dataframe(hits[cols].reset_index(drop=True))

    with st.spinner("Assembling context..."):
        context = assemble_context(hits, max_ctx_tokens=max_ctx_tokens)

    with st.expander("Context (evidence block)", expanded=False):
        st.code(context)

    with st.spinner("Generating answer..."):
        prompt = build_prompt(question, context)
        raw = generate_answer_hf(prompt, max_new_tokens=max_new_tokens)
        final = ensure_citations(raw)

    st.subheader("Answer")
    st.write(final)

    
    st.markdown("**Used citations:**")
    for c in hits["citation"].head(k).tolist():
        st.markdown(f"- {c}")