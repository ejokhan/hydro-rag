#!/bin/bash
#SBATCH -J rag-llama
#SBATCH -o slurm-llama-%j.out
#SBATCH -e slurm-llama-%j.err
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

pip install -q accelerate

# Copy the GPU script but change model name
python -c "
import sys
sys.path.insert(0, 'src/retrieval')

# Patch the model name and run
import rag_gpu
rag_gpu.main.__code__  # just checking import works
"

# Run with Llama-3.2-3B
sed 's|Qwen/Qwen2.5-3B-Instruct|meta-llama/Llama-3.2-3B-Instruct|g' src/retrieval/rag_gpu.py > /tmp/rag_llama.py
python /tmp/rag_llama.py

echo "DONE!"
