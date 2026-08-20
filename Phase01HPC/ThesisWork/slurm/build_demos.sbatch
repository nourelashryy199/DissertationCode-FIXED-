#!/bin/bash
#SBATCH --job-name=build_demos
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=build_demos_%j.out
#SBATCH --error=build_demos_%j.err

PYTHON=/users/msp25noe/.conda/envs/dissertation_env/bin/python3

$PYTHON --version
$PYTHON -c "import sys; assert sys.version_info >= (3, 10), f'Python too old: {sys.version}'"

# CONFIRM this top-level folder name matches what's actually on Stanage —
# if it's still "DissertationCode" (not "-FIXED-"), adjust this line.
export DISSERTATION_ROOT=/users/msp25noe/Thesis_Code/DissertationCode-FIXED-/Phase01HPC/ThesisWork
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache

cd $DISSERTATION_ROOT

# build_demonstrations.py was split into two independent scripts —
# demo-building (train.csv only) and eval-pool-building (test.csv only).
echo "Starting build_demo_n_shot.py at $(date)"
$PYTHON scripts/build_demo_n_shot.py
echo "Finished build_demo_n_shot.py at $(date)"

echo "Starting build_eval_pools.py at $(date)"
$PYTHON scripts/build_eval_pools.py
echo "Finished build_eval_pools.py at $(date)"
