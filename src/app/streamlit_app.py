"""
HydroRAG - Scientific Literature Q&A System
Professional Streamlit app with live LLM answers via Groq
"""
import streamlit as st
import json
import os
import numpy as np
import faiss
import time

st.set_page_config(
    page_title="HydroRAG | Scientific Q&A",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
.stApp {
    font-family: 'DM Sans', sans-serif;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    font-size: 1.05rem;
    opacity: 0.85;
    margin-top: 0.5rem;
    font-weight: 400;
}
.header-stats {
    display: flex;
    gap: 2rem;
    margin-top: 1.2rem;
}
.header-stat {
    background: rgba(255,255,255,0.1);
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    backdrop-filter: blur(10px);
}
.header-stat .num {
    font-size: 1.3rem;
    font-weight: 700;
    color: #38bdf8;
}
.header-stat .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
}

/* Search box */
.stTextInput > div > div > input {
    font-size: 1.1rem !important;
    padding: 0.8rem 1rem !important;
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0c4a6e !important;
    box-shadow: 0 0 0 3px rgba(12,74,110,0.1) !important;
}

/* Result cards */
.result-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.result-card:hover {
    border-color: #0c4a6e;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.result-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}
.result-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.8rem;
    flex-wrap: wrap;
}
.meta-tag {
    font-size: 0.78rem;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.tag-score { background: #ecfdf5; color: #065f46; }
.tag-year { background: #eff6ff; color: #1e40af; }
.tag-pubmed { background: #f0fdf4; color: #166534; }
.tag-arxiv { background: #eef2ff; color: #3730a3; }
.result-text {
    font-size: 0.9rem;
    color: #475569;
    line-height: 1.6;
    border-left: 3px solid #e2e8f0;
    padding-left: 1rem;
    margin-top: 0.8rem;
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}
.answer-box h3 {
    color: #0c4a6e;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
}
.answer-text {
    color: #1e293b;
    font-size: 0.95rem;
    line-height: 1.7;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f8fafc;
}
.sidebar-section {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.sidebar-section h4 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 0.8rem;
}

/* Metrics row */
.metric-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.3rem;
}
.metric-label { color: #64748b; font-size: 0.85rem; }
.metric-value { color: #0f172a; font-weight: 600; font-size: 0.85rem; }

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    color: #94a3b8;
    font-size: 0.85rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 3rem;
}
.footer a { color: #0c4a6e; text-decoration: none; font-weight: 500; }

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- LOAD RAG SYSTEM ----
@st.cache_resource
def load_rag_system():
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

def generate_answer(query, chunks, api_key):
    """Generate answer using Groq (free Llama-3.3 70B)"""
    from groq import Groq
    client = Groq(api_key=api_key)
    
    context = ""
    for i, chunk in enumerate(chunks):
        authors = chunk["authors"]
        if isinstance(authors, list):
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al." if authors else "Unknown"
        else:
            author_str = str(authors)
        context += f"\n[{i+1}] {chunk['title']} ({author_str}, {chunk['year']})\n"
        context += f"{chunk['text']}\n"
    
    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a scientific research assistant. Answer based ONLY on the provided papers. Cite sources using [1], [2], etc. Be concise and accurate. If the papers don't contain enough information, say so."},
            {"role": "user", "content": f"RETRIEVED PAPERS:\n{context}\n\nQUESTION: {query}\n\nANSWER:"}
        ],
        max_tokens=800,
        temperature=0.3,
    )
    gen_time = time.time() - start
    return response.choices[0].message.content, gen_time

# ---- HEADER ----
st.markdown("""
<div class="main-header">
    <h1>🌊 HydroRAG</h1>
    <p>AI-powered scientific literature Q&A across hydrology, environmental science, and healthcare</p>
    <div class="header-stats">
        <div class="header-stat">
            <div class="num">8,618</div>
            <div class="label">Papers Indexed</div>
        </div>
        <div class="header-stat">
            <div class="num">21,012</div>
            <div class="label">Searchable Chunks</div>
        </div>
        <div class="header-stat">
            <div class="num">0.809</div>
            <div class="label">MRR Score</div>
        </div>
        <div class="header-stat">
            <div class="num">Llama 3.3 70B</div>
            <div class="label">LLM via Groq</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h4>About</h4>
        <p style="font-size: 0.9rem; color: #475569; line-height: 1.5;">
        HydroRAG retrieves relevant scientific papers and generates cited answers 
        using retrieval-augmented generation (RAG). Built over 8,618 papers from 
        PubMed and ArXiv.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Groq API key input
    st.markdown("""<div class="sidebar-section"><h4>Configuration</h4></div>""", unsafe_allow_html=True)
    groq_key = st.text_input("Groq API Key (free at console.groq.com)", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    top_k = st.slider("Results to retrieve", 3, 10, 5)
    show_chunks = st.checkbox("Show retrieved text", value=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>Retrieval Metrics</h4>
        <div class="metric-row"><span class="metric-label">MRR</span><span class="metric-value">0.809</span></div>
        <div class="metric-row"><span class="metric-label">Precision@5</span><span class="metric-value">0.855</span></div>
        <div class="metric-row"><span class="metric-label">Hit Rate</span><span class="metric-value">0.938</span></div>
        <div class="metric-row"><span class="metric-label">Avg Latency</span><span class="metric-value">~8ms</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>Chunking Comparison</h4>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe({
        "Strategy": ["Sentence", "Context-Enriched", "Recursive", "Semantic", "Fixed"],
        "MRR": [0.817, 0.809, 0.728, 0.740, 0.749],
    }, hide_index=True, use_container_width=True)
    
    st.markdown("""
    <div class="sidebar-section">
        <h4>Tech Stack</h4>
        <p style="font-size: 0.82rem; color: #475569; line-height: 1.6;">
        <b>Embedding:</b> all-MiniLM-L6-v2<br>
        <b>Vector DB:</b> FAISS (21,012 vectors)<br>
        <b>LLM:</b> Llama 3.3 70B via Groq<br>
        <b>Chunking:</b> Context-Enriched<br>
        <b>Data:</b> PubMed + ArXiv
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---- LOAD SYSTEM ----
with st.spinner("Loading search engine..."):
    index, texts, metadata, model = load_rag_system()

# ---- SAMPLE QUESTIONS ----
st.markdown("**Try a question:**")
sample_cols = st.columns(3)
samples = [
    "What ML models work best for flood prediction?",
    "How is anomaly detection applied to water quality?",
    "What are transformer models used for in hydrology?",
    "How can NLP support clinical decision making?",
    "What challenges exist in streamflow forecasting?",
    "What evaluation metrics are used for RAG systems?",
]
for i, sample in enumerate(samples):
    with sample_cols[i % 3]:
        if st.button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.query = sample

# ---- SEARCH ----
query = st.text_input(
    "Ask a question about scientific literature:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What deep learning methods are used for groundwater prediction?",
    label_visibility="collapsed"
)

if query:
    # Retrieve
    with st.spinner("Searching 21,012 chunks..."):
        results, latency = retrieve(query, index, texts, metadata, model, top_k=top_k)
    
    # Generate answer if API key available
    if groq_key:
        with st.spinner("Generating answer with Llama 3.3 70B..."):
            try:
                answer, gen_time = generate_answer(query, results, groq_key)
                st.markdown(f"""
                <div class="answer-box">
                    <h3>🤖 AI Answer <span style="font-size: 0.8rem; font-weight: 400; color: #64748b;">(generated in {gen_time:.1f}s by Llama 3.3 70B)</span></h3>
                    <div class="answer-text">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Answer generation failed: {str(e)[:100]}. Showing retrieved papers below.")
    else:
        st.info("Add a free Groq API key in the sidebar to get AI-generated answers. Get one at console.groq.com")
    
    # Show results
    st.markdown(f"**Retrieved {len(results)} papers** in {latency:.0f}ms")
    
    for i, r in enumerate(results):
        authors = r["authors"]
        if isinstance(authors, list):
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al." if authors else "Unknown"
        else:
            author_str = str(authors)
        
        source_class = "tag-pubmed" if r["source"] == "pubmed" else "tag-arxiv"
        source_label = "PubMed" if r["source"] == "pubmed" else "ArXiv"
        
        chunk_html = ""
        if show_chunks:
            text_preview = r["text"][:500].replace("\n", "<br>")
            chunk_html = f'<div class="result-text">{text_preview}...</div>'
        
        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">[{i+1}] {r['title']}</div>
            <div class="result-meta">
                <span class="meta-tag tag-score">Score: {r['score']:.4f}</span>
                <span class="meta-tag tag-year">{r['year']}</span>
                <span class="meta-tag {source_class}">{source_label}</span>
            </div>
            <div style="font-size: 0.85rem; color: #64748b;">{author_str} {('| ' + r['journal']) if r.get('journal') else ''}</div>
            {chunk_html}
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #94a3b8;">
        <p style="font-size: 1.2rem;">Type a question above or click a sample to get started</p>
        <p style="font-size: 0.9rem;">Searches across hydrology, water quality, climate science, healthcare, and AI/ML literature</p>
    </div>
    """, unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("""
<div class="footer">
    Built by <a href="https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en">Ijaz Ul Haq, Ph.D.</a> | 
    <a href="https://github.com/ejokhan/hydro-rag">GitHub</a> | 
    <a href="https://linkedin.com">LinkedIn</a> | 
    University of Vermont<br>
    Powered by FAISS, Sentence-Transformers, and Llama 3.3 70B via Groq
</div>
""", unsafe_allow_html=True)
