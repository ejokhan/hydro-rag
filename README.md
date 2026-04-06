# 🌊 HydroRAG: Scientific Literature Q&A System

A production-grade Retrieval-Augmented Generation (RAG) system built over 8,618 scientific papers from PubMed and ArXiv, serving hydrology, healthcare, and AI/ML domains.

**[Live Demo](https://hydrorag-app.streamlit.app)** | **[Blog Post](coming soon)**

## What This Does

Ask a question about scientific literature and get relevant papers with similarity scores in milliseconds. The system searches across 21,012 indexed chunks using semantic similarity.

## Key Results

Systematic evaluation across 5 chunking strategies with 40 domain-expert test questions:

| Strategy | MRR | P@5 | Hit Rate | Latency |
|----------|-----|-----|----------|---------|
| Sentence | **0.817** | 0.795 | 0.896 | 8.8ms |
| Context-Enriched | 0.809 | **0.855** | **0.938** | 131.3ms |
| Fixed | 0.749 | 0.800 | 0.912 | 8.3ms |
| Semantic | 0.740 | 0.800 | 0.917 | 7.6ms |
| Recursive | 0.728 | 0.825 | 0.917 | 7.7ms |

**Finding:** Sentence-based chunking achieved the highest MRR (0.817), outperforming the industry-recommended recursive splitting (0.728) by 12.2% on scientific literature.

## Architecture
Papers (PubMed + ArXiv) → Chunking (5 strategies) → Embedding (MiniLM/BGE-M3/Qwen3)
→ FAISS Vector Index → Query → Retrieve Top-K → LLM Generation → Cited Answer

## Pipeline Components

**Data Collection:** 8,618 papers from PubMed (5,152) and ArXiv (3,466) across hydrology, healthcare, and AI/ML domains.

**Chunking Strategies:** Fixed-size, sentence-based, semantic, recursive (industry standard), and context-enriched (with metadata prefixes).

**Embedding Models:** all-MiniLM-L6-v2 (baseline), BAAI/bge-m3 (production), Qwen3-Embedding (state-of-the-art).

**Vector Database:** FAISS with normalized inner product search for cosine similarity.

**Evaluation:** 40 expert-curated test questions across 4 domains and 3 difficulty levels, measuring MRR, Precision@5, keyword hit rate, and latency.

**Frontend:** Interactive Streamlit app with real-time search and evaluation dashboard.

## Tech Stack

- Python, PyTorch, Hugging Face Transformers, Sentence-Transformers
- FAISS for vector search
- Streamlit for frontend
- NVIDIA A100/H100 GPUs on TACC Lonestar6 (HPC/SLURM)
- Distributed training infrastructure (DDP, FSDP)

## Project Structure
hydro-rag/
├── src/
│   ├── data_collection/    # Paper collection from PubMed + ArXiv
│   ├── chunking/           # 5 chunking strategies
│   ├── embedding/          # Embedding pipeline with FAISS
│   ├── retrieval/          # RAG query engine + GPU inference
│   ├── evaluation/         # Automated evaluation framework
│   └── app/                # Streamlit frontend
├── scripts/                # SLURM job scripts for HPC
├── data/
│   ├── raw/                # Collected papers
│   ├── processed/          # Chunked papers
│   ├── embeddings/         # FAISS indices + vectors
│   └── evaluation/         # Test questions + results
└── configs/

## Quick Start
```bash
# Install dependencies
pip install sentence-transformers faiss-cpu streamlit

# Collect papers
python src/data_collection/collect_papers.py --source both --max-per-query 500

# Chunk papers (all 5 strategies)
python src/chunking/chunk_papers.py --strategy all

# Embed chunks
python src/embedding/embed_chunks.py --strategy context_enriched --model all-MiniLM-L6-v2

# Run evaluation
python src/evaluation/evaluate_retrieval.py

# Launch frontend
streamlit run src/app/streamlit_app.py
```

## Author

**Ijaz Ul Haq, Ph.D.** — AI Research Scientist
University of Vermont | [Google Scholar](https://scholar.google.com) | [LinkedIn](https://linkedin.com)

Built on TACC Lonestar6 supercomputer through the NSF NAIRR Pilot program.
