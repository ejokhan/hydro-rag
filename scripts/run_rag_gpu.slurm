#!/bin/bash
#SBATCH -J hydro-rag
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH -t 01:00:00
#SBATCH -A TG-NAIRR240491

# Load CUDA first
module load cuda/12.2

# Activate environment
source /work/10655/ejokhan123/ls6/miniconda3/bin/activate
conda activate rag

export HF_HOME=/work/10655/ejokhan123/ls6/hydro-rag/.hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/10655/ejokhan123/ls6/hydro-rag

pip install accelerate -q

python src/retrieval/rag_gpu.py
