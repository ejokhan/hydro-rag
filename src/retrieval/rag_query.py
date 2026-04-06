"""
RAG Query Engine for Hydro-RAG
Retrieves relevant chunks and generates answers using an LLM.
This is the final piece — the complete RAG pipeline.
"""
import json
import os
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_index(embeddings_dir):
    """Load FAISS index, texts, and metadata."""
    index = faiss.read_index(os.path.join(embeddings_dir, "index.faiss"))
    with open(os.path.join(embeddings_dir, "texts.json"), "r") as f:
        texts = json.load(f)
    with open(os.path.join(embeddings_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print(f"Loaded index: {index.ntotal} vectors")
    return index, texts, metadata

def retrieve(query, index, texts, metadata, model, top_k=5):
    """Embed query and retrieve top-k relevant chunks."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_embedding.astype(np.float32), top_k)
    
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
    return results

def generate_answer(query, retrieved_chunks, api_key):
    """Send query + retrieved context to Gemini and get an answer."""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-lite")
    
    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        authors = chunk["authors"][0] if chunk["authors"] else "Unknown"
        context += f"\n[{i+1}] {chunk['title']} ({authors}, {chunk['year']})\n"
        context += f"{chunk['text']}\n"
    
    prompt = f"""You are a scientific research assistant. Answer the question based ONLY on the provided research papers. 
Cite your sources using [1], [2], etc. If the provided papers don't contain enough information, say so.

RETRIEVED PAPERS:
{context}

QUESTION: {query}

ANSWER (cite sources with [1], [2], etc.):"""
    
    response = model.generate_content(prompt)
    return response.text

def interactive_mode(index, texts, metadata, embed_model, api_key):
    """Interactive Q&A loop."""
    print("\n" + "=" * 60)
    print("HYDRO-RAG Interactive Query Engine")
    print("Type your question and press Enter.")
    print("Type 'quit' to exit.")
    print("=" * 60)
    
    while True:
        print()
        query = input("Question: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue
        
        # Step 1: Retrieve
        print("\nSearching 21,012 chunks...")
        results = retrieve(query, index, texts, metadata, embed_model, top_k=5)
        
        print(f"\nTop {len(results)} retrieved papers:")
        for i, r in enumerate(results):
            print(f"  [{i+1}] (score: {r['score']:.4f}) {r['title'][:70]}... ({r['year']})")
        
        # Step 2: Generate
        print("\nGenerating answer...")
        try:
            answer = generate_answer(query, results, api_key)
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer)
            print(f"{'='*60}")
        except Exception as e:
            print(f"\nError generating answer: {e}")
            print("Retrieved chunks are shown above — the retrieval part works!")

def main():
    parser = argparse.ArgumentParser(description="RAG Query Engine")
    parser.add_argument("--embeddings-dir", default="data/embeddings/context_enriched_all-MiniLM-L6-v2")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--query", type=str, help="Single query (otherwise interactive mode)")
    args = parser.parse_args()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        print("  export GEMINI_API_KEY='your-key-here'")
        return
    
    # Load everything
    print("Loading RAG system...")
    index, texts, metadata = load_index(args.embeddings_dir)
    
    print(f"Loading embedding model: {args.model}")
    embed_model = SentenceTransformer(args.model)
    
    if args.query:
        results = retrieve(args.query, index, texts, metadata, embed_model)
        answer = generate_answer(args.query, results, api_key)
        print(f"\nAnswer: {answer}")
    else:
        interactive_mode(index, texts, metadata, embed_model, api_key)

if __name__ == "__main__":
    main()
