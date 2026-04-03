"""
Embedding Script for Hydro-RAG
Converts text chunks into vectors using embedding models.
Stores vectors in FAISS index for fast retrieval.
"""
import json
import os
import argparse
import numpy as np
import faiss
import time
from sentence_transformers import SentenceTransformer

def load_chunks(processed_dir, strategy, sources=["pubmed", "arxiv"]):
    """Load chunks from processed directory."""
    all_chunks = []
    for source in sources:
        path = os.path.join(processed_dir, strategy, f"{source}_chunks.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                chunks = json.load(f)
                all_chunks.extend(chunks)
                print(f"  Loaded {len(chunks)} chunks from {source}")
    return all_chunks

def embed_chunks(chunks, model_name="all-MiniLM-L6-v2", batch_size=256):
    """
    Embed all chunks using a sentence-transformers model.
    Returns numpy array of embeddings.
    """
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # normalize for cosine similarity
    )
    elapsed = time.time() - start
    
    print(f"Done! {len(texts)} chunks embedded in {elapsed:.1f}s")
    print(f"  Speed: {len(texts)/elapsed:.1f} chunks/sec")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    
    return embeddings

def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search.
    Uses IndexFlatIP (inner product) since embeddings are normalized,
    which is equivalent to cosine similarity.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine sim for normalized vectors
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built: {index.ntotal} vectors, {dim} dimensions")
    return index

def test_retrieval(index, chunks, model, queries, top_k=5):
    """Quick test: embed a query and retrieve top-k chunks."""
    print(f"\n{'='*60}")
    print("TEST RETRIEVAL")
    print(f"{'='*60}")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        query_embedding = model.encode([query], normalize_embeddings=True)
        scores, indices = index.search(query_embedding.astype(np.float32), top_k)
        
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            chunk = chunks[idx]
            text_preview = chunk["text"][:150].replace("\n", " ")
            print(f"  [{rank+1}] Score: {score:.4f}")
            print(f"      Title: {chunk['title'][:80]}")
            print(f"      Year: {chunk['year']} | Source: {chunk['source']}")
            print(f"      Text: {text_preview}...")
            print()

def main():
    parser = argparse.ArgumentParser(description="Embed chunks for Hydro-RAG")
    parser.add_argument("--strategy", default="context_enriched")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--input-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/embeddings")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test", action="store_true", help="Run test queries after embedding")
    args = parser.parse_args()
    
    model_short = args.model.split("/")[-1]
    
    print("=" * 60)
    print("HYDRO-RAG Embedding Pipeline")
    print(f"Model: {args.model}")
    print(f"Strategy: {args.strategy}")
    print("=" * 60)
    
    # Load chunks
    print(f"\nLoading chunks (strategy: {args.strategy})...")
    chunks = load_chunks(args.input_dir, args.strategy)
    print(f"Total chunks: {len(chunks)}")
    
    # Embed
    embeddings = embed_chunks(chunks, model_name=args.model, batch_size=args.batch_size)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_dir = os.path.join(args.output_dir, f"{args.strategy}_{model_short}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))
    
    # Save embeddings
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    
    # Save chunk metadata (without text to save space, text is in processed/)
    meta = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    
    # Save chunk texts separately for retrieval display
    texts = [c["text"] for c in chunks]
    with open(os.path.join(save_dir, "texts.json"), "w") as f:
        json.dump(texts, f)
    
    print(f"\nSaved to: {save_dir}/")
    print(f"  index.faiss  - FAISS vector index")
    print(f"  embeddings.npy - raw embeddings ({embeddings.shape})")
    print(f"  metadata.json - chunk metadata")
    print(f"  texts.json - chunk texts")
    
    # Test retrieval
    if args.test:
        model = SentenceTransformer(args.model)
        test_queries = [
            "what machine learning model works best for flood prediction",
            "LSTM for streamflow forecasting",
            "anomaly detection in water quality data",
            "clinical decision support using NLP",
            "retrieval augmented generation evaluation metrics",
        ]
        test_retrieval(index, chunks, model, test_queries)

if __name__ == "__main__":
    main()
