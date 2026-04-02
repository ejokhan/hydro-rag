"""
Paper Collection Script for Hydro-RAG
Pulls papers from PubMed (abstracts) and ArXiv (abstracts + PDFs)
"""
import json
import os
import time
import argparse
from datetime import datetime

def collect_pubmed(queries, max_per_query=500, output_dir="data/raw/pubmed"):
    from Bio import Entrez, Medline
    Entrez.email = "ihaq@uvm.edu"
    os.makedirs(output_dir, exist_ok=True)
    all_papers = []
    seen_ids = set()
    for query in queries:
        print(f"\n[PubMed] Searching: '{query}'")
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_per_query, sort="relevance")
        results = Entrez.read(handle)
        handle.close()
        ids = results["IdList"]
        print(f"  Found {len(ids)} papers")
        for i in range(0, len(ids), 100):
            batch_ids = [pid for pid in ids[i:i+100] if pid not in seen_ids]
            if not batch_ids:
                continue
            handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="text")
            records = list(Medline.parse(handle))
            handle.close()
            for record in records:
                pmid = record.get("PMID", "")
                if pmid in seen_ids:
                    continue
                seen_ids.add(pmid)
                abstract = record.get("AB", "")
                if not abstract:
                    continue
                paper = {
                    "id": f"pubmed_{pmid}",
                    "source": "pubmed",
                    "pmid": pmid,
                    "title": record.get("TI", ""),
                    "abstract": abstract,
                    "authors": record.get("AU", []),
                    "journal": record.get("JT", ""),
                    "year": record.get("DP", "").split()[0] if record.get("DP") else "",
                    "keywords": record.get("MH", []),
                    "doi": record.get("AID", [""])[0] if record.get("AID") else "",
                    "collected_at": datetime.now().isoformat()
                }
                all_papers.append(paper)
            print(f"  Batch {i//100 + 1}: collected {len(all_papers)} total papers")
            time.sleep(0.5)
    output_path = os.path.join(output_dir, "pubmed_papers.json")
    with open(output_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[PubMed] Saved {len(all_papers)} papers to {output_path}")
    return all_papers

def collect_arxiv(queries, max_per_query=300, output_dir="data/raw/arxiv", download_pdfs=False):
    import arxiv
    os.makedirs(output_dir, exist_ok=True)
    if download_pdfs:
        os.makedirs(os.path.join(output_dir, "pdfs"), exist_ok=True)
    all_papers = []
    seen_ids = set()
    for query in queries:
        print(f"\n[ArXiv] Searching: '{query}'")
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_per_query, sort_by=arxiv.SortCriterion.Relevance)
        count = 0
        for result in client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            paper = {
                "id": f"arxiv_{arxiv_id}",
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "year": str(result.published.year),
                "doi": result.doi or "",
                "pdf_url": result.pdf_url,
                "collected_at": datetime.now().isoformat()
            }
            if download_pdfs:
                try:
                    pdf_path = os.path.join(output_dir, "pdfs", f"{arxiv_id.replace('/', '_')}.pdf")
                    if not os.path.exists(pdf_path):
                        result.download_pdf(dirpath=os.path.join(output_dir, "pdfs"), filename=f"{arxiv_id.replace('/', '_')}.pdf")
                        paper["pdf_path"] = pdf_path
                        time.sleep(1)
                except Exception as e:
                    print(f"  Failed PDF {arxiv_id}: {e}")
            all_papers.append(paper)
            count += 1
        print(f"  Collected {count} papers (total: {len(all_papers)})")
    output_path = os.path.join(output_dir, "arxiv_papers.json")
    with open(output_path, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"\n[ArXiv] Saved {len(all_papers)} papers to {output_path}")
    return all_papers

def main():
    parser = argparse.ArgumentParser(description="Collect papers for Hydro-RAG")
    parser.add_argument("--source", choices=["pubmed", "arxiv", "both"], default="both")
    parser.add_argument("--max-per-query", type=int, default=500)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()
    hydro_queries = [
        "streamflow prediction machine learning",
        "hydrological modeling deep learning",
        "flood forecasting neural network",
        "water quality anomaly detection",
        "rainfall runoff model",
        "groundwater level prediction",
        "river discharge time series",
        "watershed hydrology data driven",
    ]
    health_queries = [
        "clinical text mining NLP",
        "electronic health records machine learning",
        "biomedical question answering",
        "medical document retrieval",
        "clinical decision support NLP",
    ]
    llm_queries = [
        "retrieval augmented generation",
        "large language model evaluation",
        "vector database embedding retrieval",
    ]
    all_queries = hydro_queries + health_queries + llm_queries
    print("=" * 60)
    print("HYDRO-RAG Paper Collection")
    print(f"Queries: {len(all_queries)}")
    print(f"Max per query: {args.max_per_query}")
    print(f"Sources: {args.source}")
    print("=" * 60)
    total = 0
    if args.source in ["pubmed", "both"]:
        pubmed_papers = collect_pubmed(queries=all_queries, max_per_query=args.max_per_query, output_dir=os.path.join(args.output_dir, "pubmed"))
        total += len(pubmed_papers)
    if args.source in ["arxiv", "both"]:
        arxiv_papers = collect_arxiv(queries=all_queries, max_per_query=args.max_per_query // 2, output_dir=os.path.join(args.output_dir, "arxiv"), download_pdfs=args.download_pdfs)
        total += len(arxiv_papers)
    print("\n" + "=" * 60)
    print(f"DONE! Total papers collected: {total}")
    print(f"Data saved in: {args.output_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
