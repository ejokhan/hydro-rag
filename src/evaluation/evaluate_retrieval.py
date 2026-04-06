"""
Evaluation Framework for Hydro-RAG
Measures retrieval quality across chunking strategies and embedding models.
"""
import json
import os
import numpy as np
import faiss
import argparse
import time
from sentence_transformers import SentenceTransformer

def load_index(embeddings_dir):
    index = faiss.read_index(os.path.join(embeddings_dir, "index.faiss"))
    with open(os.path.join(embeddings_dir, "texts.json"), "r") as f:
        texts = json.load(f)
    with open(os.path.join(embeddings_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return index, texts, metadata

def keyword_hit_rate(retrieved_texts, expected_keywords):
    """What fraction of expected keywords appear in retrieved chunks?"""
    combined_text = " ".join(retrieved_texts).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords) if expected_keywords else 0

def mrr_score(retrieved_texts, expected_keywords):
    """Mean Reciprocal Rank: how early does the first relevant result appear?"""
    for i, text in enumerate(retrieved_texts):
        text_lower = text.lower()
        if all(kw.lower() in text_lower for kw in expected_keywords):
            return 1.0 / (i + 1)
    # Partial match fallback
    for i, text in enumerate(retrieved_texts):
        text_lower = text.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        if hits >= len(expected_keywords) * 0.5:
            return 1.0 / (i + 1)
    return 0.0

def precision_at_k(retrieved_texts, expected_keywords, k=5):
    """How many of the top-k results are relevant?"""
    relevant = 0
    for text in retrieved_texts[:k]:
        text_lower = text.lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        if hits >= len(expected_keywords) * 0.5:
            relevant += 1
    return relevant / k

def evaluate_config(embeddings_dir, questions, model_name, top_k=5):
    """Evaluate a single configuration (chunking + embedding combo)."""
    print(f"\n  Loading index from {embeddings_dir}...")
    
    if not os.path.exists(os.path.join(embeddings_dir, "index.faiss")):
        print(f"  SKIPPED: index not found")
        return None
    
    index, texts, metadata = load_index(embeddings_dir)
    model = SentenceTransformer(model_name)
    
    results = {
        "config": embeddings_dir,
        "num_vectors": index.ntotal,
        "questions": [],
        "avg_keyword_hit_rate": 0,
        "avg_mrr": 0,
        "avg_precision_at_5": 0,
        "avg_latency_ms": 0,
    }
    
    all_hit_rates = []
    all_mrr = []
    all_precision = []
    all_latency = []
    
    for q in questions:
        start = time.time()
        query_emb = model.encode([q["query"]], normalize_embeddings=True)
        scores, indices = index.search(query_emb.astype(np.float32), top_k)
        latency = (time.time() - start) * 1000
        
        retrieved_texts = [texts[idx] for idx in indices[0]]
        
        hit_rate = keyword_hit_rate(retrieved_texts, q["expected_keywords"])
        mrr = mrr_score(retrieved_texts, q["expected_keywords"])
        prec = precision_at_k(retrieved_texts, q["expected_keywords"])
        
        all_hit_rates.append(hit_rate)
        all_mrr.append(mrr)
        all_precision.append(prec)
        all_latency.append(latency)
        
        results["questions"].append({
            "id": q["id"],
            "query": q["query"],
            "domain": q["domain"],
            "difficulty": q["difficulty"],
            "keyword_hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_5": prec,
            "latency_ms": latency,
            "top_score": float(scores[0][0]),
            "top_title": metadata[indices[0][0]].get("title", "")[:80],
        })
    
    results["avg_keyword_hit_rate"] = np.mean(all_hit_rates)
    results["avg_mrr"] = np.mean(all_mrr)
    results["avg_precision_at_5"] = np.mean(all_precision)
    results["avg_latency_ms"] = np.mean(all_latency)
    
    return results

def print_results_table(all_results):
    """Print a nice comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Configuration':<50} {'Hit Rate':>10} {'MRR':>8} {'P@5':>8} {'Latency':>10}")
    print("=" * 90)
    
    for r in sorted(all_results, key=lambda x: x["avg_mrr"], reverse=True):
        config_name = r["config"].replace("data/embeddings/", "")
        print(f"{config_name:<50} {r['avg_keyword_hit_rate']:>10.3f} {r['avg_mrr']:>8.3f} {r['avg_precision_at_5']:>8.3f} {r['avg_latency_ms']:>8.1f}ms")
    
    print("=" * 90)
    
    # Breakdown by domain
    best = max(all_results, key=lambda x: x["avg_mrr"])
    print(f"\nBest config: {best['config']}")
    print(f"\nBreakdown by domain (best config):")
    domains = {}
    for q in best["questions"]:
        d = q["domain"]
        if d not in domains:
            domains[d] = {"hit_rates": [], "mrr": [], "precision": []}
        domains[d]["hit_rates"].append(q["keyword_hit_rate"])
        domains[d]["mrr"].append(q["mrr"])
        domains[d]["precision"].append(q["precision_at_5"])
    
    print(f"  {'Domain':<15} {'Hit Rate':>10} {'MRR':>8} {'P@5':>8} {'Count':>8}")
    for d, vals in domains.items():
        print(f"  {d:<15} {np.mean(vals['hit_rates']):>10.3f} {np.mean(vals['mrr']):>8.3f} {np.mean(vals['precision']):>8.3f} {len(vals['mrr']):>8}")
    
    # Breakdown by difficulty
    print(f"\nBreakdown by difficulty (best config):")
    diffs = {}
    for q in best["questions"]:
        d = q["difficulty"]
        if d not in diffs:
            diffs[d] = {"mrr": []}
        diffs[d]["mrr"].append(q["mrr"])
    
    for d in ["easy", "medium", "hard"]:
        if d in diffs:
            print(f"  {d:<10} MRR: {np.mean(diffs[d]['mrr']):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval")
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    
    # Load test questions
    with open("data/evaluation/test_questions.json", "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} test questions")
    
    # Find all available configurations
    embeddings_base = args.embeddings_dir
    configs = []
    if os.path.exists(embeddings_base):
        for d in os.listdir(embeddings_base):
            full_path = os.path.join(embeddings_base, d)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "index.faiss")):
                configs.append(full_path)
    
    if not configs:
        print("No embedding configs found! Run embed_chunks.py first.")
        return
    
    print(f"Found {len(configs)} configurations to evaluate")
    
    # Evaluate each config
    all_results = []
    for config_path in sorted(configs):
        print(f"\nEvaluating: {config_path}")
        model_name = args.model
        # Try to infer model from directory name
        if "MiniLM" in config_path or "minilm" in config_path.lower():
            model_name = "all-MiniLM-L6-v2"
        
        result = evaluate_config(config_path, questions, model_name)
        if result:
            all_results.append(result)
    
    # Print comparison
    print_results_table(all_results)
    
    # Save full results
    os.makedirs("data/evaluation", exist_ok=True)
    output_path = "data/evaluation/retrieval_results.json"
    
    # Remove non-serializable items
    for r in all_results:
        for q in r["questions"]:
            for k, v in q.items():
                if isinstance(v, np.floating):
                    q[k] = float(v)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")

if __name__ == "__main__":
    main()
