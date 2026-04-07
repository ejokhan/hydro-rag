#!/bin/bash
#SBATCH -J rag-mistral
#SBATCH -o slurm-mistral-%j.out
#SBATCH -e slurm-mistral-%j.err
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

pip install -q accelerate

sed 's|Qwen/Qwen2.5-3B-Instruct|mistralai/Mistral-7B-Instruct-v0.3|g' src/retrieval/rag_gpu.py > /tmp/rag_mistral.py
python /tmp/rag_mistral.py

echo "DONE!"
