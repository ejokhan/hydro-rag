"""
HydroRAG - Scientific Literature Q&A System
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

.stApp { font-family: 'Source Sans 3', sans-serif; }

div[data-testid="stMainBlockContainer"] { max-width: 1100px; margin: 0 auto; }

.hero {
    background: #0B1A2E;
    padding: 2rem 2.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem; }
.hero p { font-size: 0.95rem; color: #94A3B8; margin: 0; }
.hero-stats {
    display: flex; gap: 1.5rem; margin-top: 1rem; flex-wrap: wrap;
}
.hero-stat {
    background: rgba(255,255,255,0.07);
    padding: 0.5rem 1rem;
    border-radius: 8px;
}
.hero-stat .val { font-size: 1.15rem; font-weight: 700; color: #38BDF8; }
.hero-stat .lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.8px; color: #64748B; }

.answer-card {
    background: #F0F9FF;
    border-left: 4px solid #0284C7;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0 1.5rem;
}
.answer-card h4 { margin: 0 0 0.5rem; color: #0369A1; font-size: 0.95rem; }
.answer-card .text { color: #1E293B; font-size: 0.92rem; line-height: 1.65; }
.answer-card .meta { font-size: 0.78rem; color: #64748B; margin-top: 0.5rem; }

.paper-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.15s;
}
.paper-card:hover { border-color: #0284C7; }
.paper-title { font-size: 0.95rem; font-weight: 600; color: #0F172A; line-height: 1.35; }
.paper-meta { display: flex; gap: 0.5rem; margin-top: 0.4rem; flex-wrap: wrap; }
.tag {
    font-size: 0.72rem; padding: 0.15rem 0.55rem; border-radius: 5px;
    font-weight: 500; font-family: 'SF Mono', 'Fira Code', monospace;
}
.tag-score { background: #ECFDF5; color: #065F46; }
.tag-year { background: #EFF6FF; color: #1E40AF; }
.tag-pm { background: #F0FDF4; color: #166534; }
.tag-ax { background: #EEF2FF; color: #3730A3; }
.paper-text {
    font-size: 0.82rem; color: #64748B; line-height: 1.55;
    border-left: 2px solid #E2E8F0; padding-left: 0.8rem;
    margin-top: 0.6rem;
}
.paper-authors { font-size: 0.8rem; color: #64748B; margin-top: 0.3rem; }

.sample-btn button {
    font-size: 0.82rem !important;
    padding: 0.4rem 0.8rem !important;
    border-radius: 8px !important;
}

.footer-text {
    text-align: center; padding: 1.5rem 0; color: #94A3B8;
    font-size: 0.82rem; border-top: 1px solid #E2E8F0; margin-top: 2rem;
}
.footer-text a { color: #0284C7; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

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
            "text": texts[idx], "score": float(score),
            "title": metadata[idx].get("title", ""),
            "authors": metadata[idx].get("authors", []),
            "year": metadata[idx].get("year", ""),
            "source": metadata[idx].get("source", ""),
            "journal": metadata[idx].get("journal", ""),
        })
    return results, latency

def generate_answer(query, chunks, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    context = ""
    for i, chunk in enumerate(chunks):
        authors = chunk["authors"]
        if isinstance(authors, list):
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al." if authors else "Unknown"
        else:
            author_str = str(authors)
        context += f"\n[{i+1}] {chunk['title']} ({author_str}, {chunk['year']})\n{chunk['text']}\n"
    start = time.time()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a scientific research assistant. Answer based ONLY on the provided papers. Cite using [1], [2], etc. Be concise and accurate."},
            {"role": "user", "content": f"PAPERS:\n{context}\n\nQUESTION: {query}\n\nANSWER:"}
        ],
        max_tokens=600, temperature=0.3,
    )
    return response.choices[0].message.content, time.time() - start

# ---- HERO ----
st.markdown("""
<div class="hero">
    <h1>🌊 HydroRAG</h1>
    <p>Ask any question about scientific literature. Get AI-generated answers with citations.</p>
    <div class="hero-stats">
        <div class="hero-stat"><div class="val">8,618</div><div class="lbl">Papers</div></div>
        <div class="hero-stat"><div class="val">21,012</div><div class="lbl">Indexed chunks</div></div>
        <div class="hero-stat"><div class="val">0.809</div><div class="lbl">MRR score</div></div>
        <div class="hero-stat"><div class="val">Llama 3.3 70B</div><div class="lbl">Answer engine</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    HydroRAG searches 8,618 scientific papers from PubMed and ArXiv
    and generates cited answers using Llama 3.3 70B.
    
    Built by **Ijaz Ul Haq, Ph.D.**  
    University of Vermont
    """)
    st.divider()
    st.markdown("### Settings")
    top_k = st.slider("Results to retrieve", 3, 10, 5)
    show_text = st.checkbox("Show retrieved text", value=True)
    
    st.divider()
    st.markdown("### Evaluation metrics")
    mc1, mc2 = st.columns(2)
    mc1.metric("MRR", "0.809")
    mc2.metric("P@5", "0.855")
    mc3, mc4 = st.columns(2)
    mc3.metric("Hit rate", "0.938")
    mc4.metric("Latency", "~8ms")
    
    st.divider()
    st.markdown("### Chunking comparison")
    st.dataframe({
        "Strategy": ["Sentence", "Context-enriched", "Recursive", "Semantic", "Fixed"],
        "MRR": [0.817, 0.809, 0.728, 0.740, 0.749],
        "P@5": [0.795, 0.855, 0.825, 0.800, 0.800],
    }, hide_index=True, use_container_width=True)
    
    st.divider()
    st.markdown("### Tech stack")
    st.caption("Embedding: all-MiniLM-L6-v2")
    st.caption("Vector DB: FAISS (21,012 vectors)")
    st.caption("LLM: Llama 3.3 70B via Groq")
    st.caption("Data: PubMed + ArXiv")
    st.caption("Infra: TACC Lonestar6 A100/H100")

# ---- LOAD ----
with st.spinner("Loading search engine..."):
    index, texts, metadata, model = load_rag_system()

# ---- SAMPLES ----
st.markdown("**Try a question:**")
cols = st.columns(3)
samples = [
    "What ML models work best for flood prediction?",
    "How is anomaly detection applied to water quality?",
    "Transformer models in streamflow forecasting",
    "NLP for clinical decision support",
    "Challenges in deep learning for hydrology",
    "Evaluation metrics for RAG systems",
]
for i, s in enumerate(samples):
    with cols[i % 3]:
        if st.button(s, key=f"s{i}", use_container_width=True):
            st.session_state.query = s

# ---- SEARCH ----
query = st.text_input(
    "Ask a question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What deep learning methods are used for groundwater prediction?",
)

if query:
    with st.spinner("Searching 21,012 chunks..."):
        results, latency = retrieve(query, index, texts, metadata, model, top_k=top_k)
    
    # Generate answer
    groq_key = os.environ.get("GROQ_API_KEY", "") or st.secrets.get("GROQ_API_KEY", "")
    if groq_key:
        with st.spinner("Generating answer with Llama 3.3 70B..."):
            try:
                answer, gen_time = generate_answer(query, results, groq_key)
                st.markdown(f"""
                <div class="answer-card">
                    <h4>AI answer</h4>
                    <div class="text">{answer}</div>
                    <div class="meta">Generated in {gen_time:.1f}s by Llama 3.3 70B via Groq</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Answer generation failed: {str(e)[:100]}")
    
    st.markdown(f"**{len(results)} papers retrieved** in {latency:.0f}ms")
    
    for i, r in enumerate(results):
        authors = r["authors"]
        if isinstance(authors, list):
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al." if authors else "Unknown"
        else:
            author_str = str(authors)
        
        src_class = "tag-pm" if r["source"] == "pubmed" else "tag-ax"
        src_label = "PubMed" if r["source"] == "pubmed" else "ArXiv"
        
        text_html = ""
        if show_text:
            preview = r["text"][:400].replace("\n", " ")
            text_html = f'<div class="paper-text">{preview}...</div>'
        
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">[{i+1}] {r['title']}</div>
            <div class="paper-meta">
                <span class="tag tag-score">{r['score']:.4f}</span>
                <span class="tag tag-year">{r['year']}</span>
                <span class="tag {src_class}">{src_label}</span>
            </div>
            <div class="paper-authors">{author_str}{(' — ' + r['journal']) if r.get('journal') else ''}</div>
            {text_html}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #94A3B8;">
        <p style="font-size: 1.1rem;">Ask a question about scientific literature</p>
        <p style="font-size: 0.85rem;">Covers hydrology, water quality, climate science, healthcare, and AI/ML</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-text">
    Built by <a href="https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en">Ijaz Ul Haq, Ph.D.</a> | 
    <a href="https://github.com/ejokhan/hydro-rag">GitHub</a> | 
    <a href="https://linkedin.com">LinkedIn</a><br>
    University of Vermont | Powered by FAISS + Llama 3.3 70B via Groq
</div>
""", unsafe_allow_html=True)
