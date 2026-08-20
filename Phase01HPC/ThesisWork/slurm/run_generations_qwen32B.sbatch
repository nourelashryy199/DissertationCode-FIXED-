#!/bin/bash
#SBATCH --job-name=run_generations_qwen32B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=4-00:00:00
#SBATCH --mem=128G
#SBATCH --output=run_generations_qwen32B_%j.out
#SBATCH --error=run_generations_qwen32B_%j.err

PYTHON=/users/msp25noe/.conda/envs/dissertation_env/bin/python3

$PYTHON --version
$PYTHON -c "import sys; assert sys.version_info >= (3, 10), f'Python too old: {sys.version}'"

export DISSERTATION_ROOT=/users/msp25noe/Thesis_Code/DissertationCode-FIXED-/Phase01HPC/ThesisWork
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache

cd $DISSERTATION_ROOT

echo "Starting results_generation.py at $(date)"
$PYTHON -u scripts/results_generation.py --model Qwen/Qwen2.5-32B-Instruct --sample_size 45
echo "Finished at $(date)"
