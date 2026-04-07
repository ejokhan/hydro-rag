# 🌊 HydroRAG: Scientific Literature Q&A System

**[Try the Live Demo](https://hydrorag.streamlit.app)** | Ask any question about hydrology, water quality, or environmental science and get AI-generated answers with citations from 8,618 research papers.

## What This Does

HydroRAG is a retrieval-augmented generation (RAG) system that searches across 21,012 indexed chunks from scientific papers and generates cited answers using Llama 3.3 70B. Ask a question in plain English, get a research-backed answer in seconds.

**Example:** "What machine learning models are most effective for flood prediction?"

The system retrieves the 5 most relevant paper sections, then an LLM reads them and writes a cited answer referencing specific studies, methods, and results.

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

    User Question
        |
        v
    Embedding Model (MiniLM) converts to vector
        |
        v
    FAISS searches 21,012 chunks for top matches
        |
        v
    Top 5 relevant paper sections retrieved
        |
        v
    LLM (Llama 3.3 70B via Groq) reads papers
        |
        v
    Cited answer with references [1], [2], etc.

## Pipeline Components

**Data:** 8,618 papers from PubMed (5,152) and ArXiv (3,466) across hydrology, water quality, climate science, healthcare, and AI/ML.

**Chunking:** 5 strategies implemented and benchmarked: fixed-size, sentence-based, semantic, recursive, and context-enriched with metadata prefixes.

**Embedding:** all-MiniLM-L6-v2 (baseline). BGE-M3 and Qwen3-Embedding evaluations in progress.

**Vector Search:** FAISS with normalized inner product for cosine similarity. Sub-10ms retrieval.

**Generation:** Llama 3.3 70B via Groq for real-time cited answers.

**Evaluation:** 40 expert-curated questions across 4 domains (hydrology, healthcare, LLM/RAG, cross-domain) and 3 difficulty levels.

## Tech Stack

Python, PyTorch, Hugging Face Transformers, Sentence-Transformers, FAISS, Groq API, Streamlit, NVIDIA A100/H100 GPUs on TACC Lonestar6 via NSF NAIRR Pilot

## Quick Start

    git clone https://github.com/ejokhan/hydro-rag.git
    cd hydro-rag
    pip install -r requirements.txt

    # Collect papers
    python src/data_collection/collect_papers.py --source both --max-per-query 500

    # Chunk (all 5 strategies)
    python src/chunking/chunk_papers.py --strategy all

    # Embed
    python src/embedding/embed_chunks.py --strategy context_enriched --model all-MiniLM-L6-v2

    # Evaluate
    python src/evaluation/evaluate_retrieval.py

    # Launch app
    streamlit run src/app/streamlit_app.py

## Author

**Ijaz Ul Haq, Ph.D.** | AI Research Scientist

University of Vermont | Water Resources Institute

[Google Scholar](https://scholar.google.com/citations?user=qHTMlKIAAAAJ&hl=en) | [LinkedIn](https://linkedin.com) | [GitHub](https://github.com/ejokhan)

Built on TACC Lonestar6 supercomputer through the NSF NAIRR Pilot program.

## License

MIT
