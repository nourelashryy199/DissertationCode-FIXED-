#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --output=smoke_test_%j.out
#SBATCH --error=smoke_test_%j.err

PYTHON=/users/msp25noe/.conda/envs/dissertation_env/bin/python3

$PYTHON --version
$PYTHON -c "import sys; assert sys.version_info >= (3, 10), f'Python too old: {sys.version}'"

export DISSERTATION_ROOT=/users/msp25noe/Thesis_Code/DissertationCode-FIXED-/Phase01HPC/ThesisWork
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache

cd /users/msp25noe/Thesis_Code/DissertationCode-FIXED-/Phase00Testing/scripts

echo "Starting smoke_test_generation.py at $(date)"
$PYTHON smoke_test_generation.py --model Qwen/Qwen2.5-7B-Instruct
echo "Finished at $(date)"
