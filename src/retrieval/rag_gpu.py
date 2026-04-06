"""
RAG with Open-Source LLM on GPU
Uses Qwen2.5-3B-Instruct - well supported, good quality
"""
import json
import os
import numpy as np
import faiss
import torch
import time
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_index(embeddings_dir):
    index = faiss.read_index(os.path.join(embeddings_dir, "index.faiss"))
    with open(os.path.join(embeddings_dir, "texts.json"), "r") as f:
        texts = json.load(f)
    with open(os.path.join(embeddings_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    print(f"Loaded FAISS index: {index.ntotal} vectors")
    return index, texts, metadata

def retrieve(query, index, texts, metadata, model, top_k=5):
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
        })
    return results

def build_prompt(query, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        authors = chunk["authors"][0] if chunk["authors"] else "Unknown"
        context += f"\n[{i+1}] {chunk['title']} ({authors}, {chunk['year']})\n"
        context += f"{chunk['text']}\n"
    prompt = f"""You are a scientific research assistant. Answer the question based ONLY on the provided research papers.
Cite your sources using [1], [2], etc. If the papers don't contain enough information, say so.
Be concise but thorough.

RETRIEVED PAPERS:
{context}

QUESTION: {query}

ANSWER:"""
    return prompt

def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected! Will run on CPU (slow)")

    # Load retrieval system
    print("\n[1/3] Loading retrieval system...")
    index, texts, metadata = load_index("data/embeddings/context_enriched_all-MiniLM-L6-v2")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load LLM
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    print(f"\n[2/3] Loading LLM ({model_name})...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    print(f"Model loaded in {time.time()-start:.1f}s")

    # Test queries
    queries = [
        "What machine learning models are most effective for flood prediction?",
        "How is anomaly detection applied to water quality monitoring?",
        "What are the main challenges in streamflow forecasting using deep learning?",
        "How can NLP be used for clinical decision support in healthcare?",
        "What evaluation metrics are used for retrieval augmented generation systems?",
    ]

    print("\n[3/3] Running RAG queries...")
    results_log = []

    for query in queries:
        print(f"\n{'='*70}")
        print(f"QUESTION: {query}")
        print(f"{'='*70}")

        retrieved = retrieve(query, index, texts, metadata, embed_model, top_k=5)
        print(f"\nRetrieved {len(retrieved)} chunks:")
        for i, r in enumerate(retrieved):
            print(f"  [{i+1}] (score: {r['score']:.4f}) {r['title'][:70]}... ({r['year']})")

        prompt = build_prompt(query, retrieved)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
            )
        gen_time = time.time() - start

        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        print(f"\nANSWER (generated in {gen_time:.1f}s):")
        print("-" * 50)
        print(answer.strip())
        print("-" * 50)

        results_log.append({
            "query": query,
            "retrieved": [{"title": r["title"], "score": r["score"], "year": r["year"]} for r in retrieved],
            "answer": answer.strip(),
            "generation_time": gen_time,
        })

    with open("data/rag_results.json", "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nResults saved to data/rag_results.json")
    print("DONE!")

if __name__ == "__main__":
    main()
