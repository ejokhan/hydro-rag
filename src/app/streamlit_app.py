"""
HydroRAG - Scientific Literature Q&A System
Streamlit frontend for the RAG pipeline.
"""
import streamlit as st
import json
import os
import numpy as np
import faiss
import time

st.set_page_config(
    page_title="HydroRAG",
    page_icon="🌊",
    layout="wide"
)

@st.cache_resource
def load_rag_system():
    """Load FAISS index, texts, metadata, and embedding model."""
    from sentence_transformers import SentenceTransformer
    
    embeddings_dir = "data/embeddings/context_enriched_all-MiniLM-L6-v2"
    
    index = faiss.read_index(os.path.join(embeddings_dir, "index.faiss"))
    with open(os.path.join(embeddings_dir, "texts.json"), "r") as f:
        texts = json.load(f)
    with open(os.path.join(embeddings_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return index, texts, metadata, model

def retrieve(query, index, texts, metadata, model, top_k=5):
    """Retrieve top-k relevant chunks."""
    start = time.time()
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)
    latency = (time.time() - start) * 1000
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "text": texts[idx],
            "score": float(score),
            "title": metadata[idx].get("title", ""),
            "authors": metadata[idx].get("authors", []),
            "year": metadata[idx].get("year", ""),
            "source": metadata[idx].get("source", ""),
            "journal": metadata[idx].get("journal", ""),
        })
    return results, latency

# --- UI ---
st.title("🌊 HydroRAG")
st.markdown("**Scientific Literature Q&A System** | 8,618 papers | 21,012 indexed chunks")
st.markdown("Search across hydrology, healthcare, and AI/ML research papers from PubMed and ArXiv.")

st.divider()

# Sidebar with system info
with st.sidebar:
    st.header("System Info")
    st.markdown("""
    **Papers:** 8,618 (PubMed + ArXiv)
    
    **Chunking:** Context-Enriched
    
    **Embedding:** all-MiniLM-L6-v2
    
    **Vector DB:** FAISS (21,012 vectors)
    
    **Dimensions:** 384
    """)
    
    st.divider()
    st.header("Settings")
    top_k = st.slider("Number of results", 1, 10, 5)
    
    st.divider()
    st.header("Evaluation Results")
    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | MRR | 0.809 |
    | P@5 | 0.855 |
    | Hit Rate | 0.938 |
    """)
    
    st.divider()
    st.header("Sample Questions")
    sample_questions = [
        "What ML models work best for flood prediction?",
        "How is anomaly detection applied to water quality?",
        "What are transformer models used for in streamflow forecasting?",
        "How can NLP support clinical decision making?",
        "What evaluation metrics are used for RAG systems?",
    ]
    for q in sample_questions:
        if st.button(q, key=q):
            st.session_state.query = q

# Load system
with st.spinner("Loading RAG system..."):
    index, texts, metadata, model = load_rag_system()

# Query input
query = st.text_input(
    "Ask a question about scientific literature:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What machine learning models are used for streamflow prediction?"
)

if query:
    with st.spinner("Searching 21,012 chunks..."):
        results, latency = retrieve(query, index, texts, metadata, model, top_k=top_k)
    
    st.markdown(f"**Found {len(results)} results in {latency:.1f}ms**")
    st.divider()
    
    for i, r in enumerate(results):
        authors = r["authors"]
        if isinstance(authors, list):
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al." if authors else "Unknown"
        else:
            author_str = str(authors)
        
        source_badge = "🟢 PubMed" if r["source"] == "pubmed" else "🔵 ArXiv"
        
        with st.expander(
            f"[{i+1}] {r['title'][:100]} ({r['year']}) — Score: {r['score']:.4f}",
            expanded=(i == 0)
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Similarity Score", f"{r['score']:.4f}")
            col2.metric("Year", r["year"])
            col3.metric("Source", source_badge)
            
            st.markdown(f"**Authors:** {author_str}")
            if r.get("journal"):
                st.markdown(f"**Journal:** {r['journal']}")
            
            st.divider()
            st.markdown("**Retrieved Text:**")
            st.markdown(r["text"])

    # Show comparison table at bottom
    st.divider()
    st.subheader("Chunking Strategy Comparison (MiniLM)")
    
    comparison_data = {
        "Strategy": ["Sentence", "Context-Enriched", "Fixed", "Semantic", "Recursive"],
        "MRR": [0.817, 0.809, 0.749, 0.740, 0.728],
        "P@5": [0.795, 0.855, 0.800, 0.800, 0.825],
        "Hit Rate": [0.896, 0.938, 0.912, 0.917, 0.917],
        "Latency (ms)": [8.8, 131.3, 8.3, 7.6, 7.7],
    }
    st.dataframe(comparison_data, use_container_width=True)

else:
    st.info("Type a question above or click a sample question in the sidebar to get started.")

st.divider()
st.markdown(
    "Built by **Ijaz Ul Haq, Ph.D.** | "
    "[GitHub](https://github.com/ijaz-ul-haq) | "
    "[Google Scholar](https://scholar.google.com) | "
    "[LinkedIn](https://linkedin.com)"
)
