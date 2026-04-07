#!/bin/bash
#SBATCH -J rag-bge
#SBATCH -o slurm-bge-%j.out
#SBATCH -e slurm-bge-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -A TG-NAIRR240491

module load cuda/12.0
source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag
export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache
export TOKENIZERS_PARALLELISM=false
cd /work/10655/ejokhan123/ls6/hydro-rag

echo "=== BGE-M3 Embeddings ==="

for STRATEGY in fixed sentence semantic recursive context_enriched; do
    echo "Embedding strategy: $STRATEGY with BGE-M3"
    python src/embedding/embed_chunks.py --strategy $STRATEGY --model BAAI/bge-m3 --batch-size 64
done

echo "=== Running evaluation ==="
python src/evaluation/evaluate_retrieval.py --model BAAI/bge-m3

echo "DONE!"
