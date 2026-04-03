"""
Chunking Script for Hydro-RAG
Five strategies: fixed-size, sentence, semantic, recursive, context-enriched
"""
import json
import os
import re
import argparse
from collections import Counter

def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_fixed_size(text, chunk_size=200, overlap=30, **kwargs):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def chunk_sentence_based(text, max_sentences=5, min_sentences=2, **kwargs):
    sentences = split_sentences(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences - 1):
        group = sentences[i:i + max_sentences]
        if len(group) >= min_sentences:
            chunks.append(" ".join(group))
        elif chunks:
            chunks[-1] += " " + " ".join(group)
    if not chunks and sentences:
        chunks.append(" ".join(sentences))
    return chunks

def chunk_semantic_sections(text, title="", **kwargs):
    chunks = []
    if len(text.split()) < 300:
        prefix = f"Title: {title}\n\n" if title else ""
        chunks.append(prefix + text)
        return chunks
    sections = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*:)', text)
    current_chunk = f"Title: {title}\n\n" if title else ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len((current_chunk + " " + section).split()) > 400:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = f"Title: {title}\n\n{section}" if title else section
        else:
            current_chunk += " " + section
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    if not chunks:
        prefix = f"Title: {title}\n\n" if title else ""
        chunks.append(prefix + text)
    return chunks

def chunk_recursive(text, chunk_size=512, overlap_pct=0.15, **kwargs):
    """
    Strategy 4: Recursive Character Splitting (INDUSTRY STANDARD 2026)
    Tries to split on paragraphs first, then sentences, then words.
    This is what LangChain's RecursiveCharacterTextSplitter does.
    512 tokens with 10-20% overlap is the recommended starting point.
    """
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]
    overlap = int(chunk_size * overlap_pct)
    
    def recursive_split(text, seps):
        if len(text.split()) <= chunk_size:
            return [text] if text.strip() else []
        
        # Try each separator in order
        for i, sep in enumerate(seps):
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                
                for part in parts:
                    candidate = (current + sep + part).strip() if current else part.strip()
                    if len(candidate.split()) > chunk_size and current:
                        chunks.append(current.strip())
                        # Keep overlap from end of previous chunk
                        overlap_words = current.split()[-overlap:] if overlap > 0 else []
                        current = " ".join(overlap_words) + sep + part if overlap_words else part
                    else:
                        current = candidate
                
                if current.strip():
                    chunks.append(current.strip())
                
                # If any chunk is still too big, split it with next separator
                if i + 1 < len(seps):
                    final = []
                    for c in chunks:
                        if len(c.split()) > chunk_size * 1.2:
                            final.extend(recursive_split(c, seps[i+1:]))
                        else:
                            final.append(c)
                    return final
                return chunks
        
        # Fallback: hard split by words
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            start = end - overlap
        return chunks
    
    result = recursive_split(text, separators)
    return [c for c in result if c.strip()]

def chunk_context_enriched(text, title="", authors=None, year="", journal="", **kwargs):
    """
    Strategy 5: Context-Enriched Chunking (STATE OF THE ART 2026)
    Each chunk gets a context prefix with metadata so it carries
    meaning even when retrieved in isolation. This is what
    production RAG systems at top companies are doing now.
    """
    # Build context prefix from metadata
    context_parts = []
    if title:
        context_parts.append(f"Paper: {title}")
    if authors and len(authors) > 0:
        author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
        context_parts.append(f"Authors: {author_str}")
    if year:
        context_parts.append(f"Year: {year}")
    if journal:
        context_parts.append(f"Journal: {journal}")
    
    context_prefix = " | ".join(context_parts)
    if context_prefix:
        context_prefix = f"[{context_prefix}]\n"
    
    # Use sentence-based splitting for the content
    sentences = split_sentences(text)
    chunks = []
    max_sentences = 5
    
    for i in range(0, len(sentences), max_sentences - 1):
        group = sentences[i:i + max_sentences]
        if len(group) >= 2:
            chunk_text = context_prefix + " ".join(group)
            chunks.append(chunk_text)
        elif chunks:
            # Append to last chunk
            chunks[-1] += " " + " ".join(group)
    
    if not chunks and sentences:
        chunks.append(context_prefix + " ".join(sentences))
    
    return chunks

def process_papers(input_path, output_dir, strategy="all"):
    print(f"\nLoading papers from {input_path}...")
    with open(input_path, "r") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")
    
    strategies = {
        "fixed": chunk_fixed_size,
        "sentence": chunk_sentence_based,
        "semantic": chunk_semantic_sections,
        "recursive": chunk_recursive,
        "context_enriched": chunk_context_enriched,
    }
    if strategy != "all":
        strategies = {strategy: strategies[strategy]}
    
    for strat_name, strat_func in strategies.items():
        print(f"\n{'='*50}")
        print(f"Chunking with strategy: {strat_name}")
        print(f"{'='*50}")
        all_chunks = []
        for paper in papers:
            text = paper.get("abstract", "")
            if not text:
                continue
            title = paper.get("title", "")
            raw_chunks = strat_func(
                text,
                title=title,
                authors=paper.get("authors", []),
                year=paper.get("year", ""),
                journal=paper.get("journal", ""),
            )
            for i, chunk_text in enumerate(raw_chunks):
                chunk = {
                    "chunk_id": f"{paper['id']}_{strat_name}_chunk_{i}",
                    "paper_id": paper["id"],
                    "source": paper.get("source", ""),
                    "title": title,
                    "authors": paper.get("authors", []),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "chunk_index": i,
                    "total_chunks": 0,
                    "strategy": strat_name,
                    "text": chunk_text,
                    "word_count": len(chunk_text.split()),
                    "char_count": len(chunk_text),
                }
                all_chunks.append(chunk)
        paper_chunk_counts = Counter(c["paper_id"] for c in all_chunks)
        for chunk in all_chunks:
            chunk["total_chunks"] = paper_chunk_counts[chunk["paper_id"]]
        strat_dir = os.path.join(output_dir, strat_name)
        os.makedirs(strat_dir, exist_ok=True)
        source_name = os.path.basename(input_path).replace("_papers.json", "")
        output_path = os.path.join(strat_dir, f"{source_name}_chunks.json")
        with open(output_path, "w") as f:
            json.dump(all_chunks, f, indent=2)
        word_counts = [c["word_count"] for c in all_chunks]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        print(f"  Papers processed: {len(papers)}")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Avg words/chunk: {avg_words:.1f}")
        print(f"  Min words: {min(word_counts) if word_counts else 0}")
        print(f"  Max words: {max(word_counts) if word_counts else 0}")
        print(f"  Chunks per paper: {len(all_chunks)/len(papers):.1f}")
        print(f"  Saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Chunk papers for Hydro-RAG")
    parser.add_argument("--strategy", choices=["fixed", "sentence", "semantic", "recursive", "context_enriched", "all"], default="all")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    print("=" * 60)
    print("HYDRO-RAG Chunking Pipeline")
    print(f"Strategy: {args.strategy}")
    print("=" * 60)
    pubmed_path = os.path.join(args.input_dir, "pubmed", "pubmed_papers.json")
    if os.path.exists(pubmed_path):
        process_papers(pubmed_path, args.output_dir, strategy=args.strategy)
    arxiv_path = os.path.join(args.input_dir, "arxiv", "arxiv_papers.json")
    if os.path.exists(arxiv_path):
        process_papers(arxiv_path, args.output_dir, strategy=args.strategy)
    print("\n" + "=" * 60)
    print("CHUNKING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
