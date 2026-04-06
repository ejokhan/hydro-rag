#!/bin/bash
#SBATCH -J rag-qwen-emb
#SBATCH -o slurm-qwen-emb-%j.out
#SBATCH -e slurm-qwen-emb-%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -A TG-NAIRR240491

module load cuda/12.2
source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag
export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache
export TOKENIZERS_PARALLELISM=false
cd /work/10655/ejokhan123/ls6/hydro-rag

echo "=== Qwen3-Embedding ==="

pip install -q accelerate

for STRATEGY in fixed sentence semantic recursive context_enriched; do
    echo "Embedding strategy: $STRATEGY with Qwen3-Embedding"
    python src/embedding/embed_chunks.py --strategy $STRATEGY --model Alibaba-NLP/gte-Qwen2-7B-instruct --batch-size 16
done

echo "=== Running evaluation ==="
python src/evaluation/evaluate_retrieval.py --model Alibaba-NLP/gte-Qwen2-7B-instruct

echo "DONE!"
