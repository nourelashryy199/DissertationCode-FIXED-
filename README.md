# Consistency Matters: A Risk-Adjusted Approach to Evaluating Prompt Strategies for Legal Reasoning

Code and experimental pipeline for the MSc Advanced Computer Science dissertation:

**“Consistency Matters: A Risk-Adjusted Approach to Evaluating Prompt Strategies for Legal Reasoning”**

University of Sheffield, 2026.

This project investigates the reliability of prompting strategies across different categories of legal reasoning.

Rather than evaluating prompting strategies using mean accuracy alone, the study evaluates performance relative to a zero-shot baseline while accounting for variation across repeated sampling runs and instruction rephrasings. It introduces a family of risk-adjusted reliability measures — including Run, Rephrasing, and Joint Fit Scores — together with Transfer Cost and Strategy Dispersion.

The central objective is to establish a risk-adjusted, category-level mapping between prompting strategies and the categories of legal reasoning they are best suited to, and to quantify the cost of applying a strategy outside the category for which it performs best.

## Repository Structure

```text
DissertationCode-FIXED-/
│
├── preparations/
│   ├── DowloadingALLLegalBench.py
│   ├── ExtractingCandidateTasks.py
│   ├── SelectingClassificationTasks.py
│   ├── thesis_test.py
│   └── data/
│
├── Phase01HPC/
│   └── ThesisWork/
│       ├── config.py
│       ├── schema.py
│       ├── model.py
│       ├── strategy_functions.py
│       ├── requirements.txt
│       ├── data/
│       ├── scripts/
│       ├── slurm/
│       ├── demonstrations/
│       ├── eval_pools/
│       ├── outputs/
│       ├── results/
│       └── figures/
│
├── Phase00Testing/
│   ├── scripts/
│   └── slurm/
│
└── Evaluation/
```

- `preparations/` contains the LegalBench download and task-selection pipeline.
- `Phase01HPC/ThesisWork/` contains the main experimental pipeline and files used for execution on Stanage.
- `Phase00Testing/` contains optional validation and smoke-testing utilities.
- `Evaluation/` contains the parsing, metric computation, statistical analysis, and visualisation scripts.

---

# Reproducing the Experiments

The full generation experiments were run on **Stanage**, the University of Sheffield HPC cluster.

The instructions below assume that the user has access to Stanage and is starting from a fresh login.

## 1. Clone the Repository

On a Stanage login node:

```bash
cd "$HOME"

git clone https://github.com/nourelashryy199/DissertationCode-FIXED-.git

cd DissertationCode-FIXED-
```

Define the repository locations:

```bash
export REPO_ROOT="$HOME/DissertationCode-FIXED-"
export DISSERTATION_ROOT="$REPO_ROOT/Phase01HPC/ThesisWork"
```

These variables should be set again after starting a new Stanage login session.

---

## 2. Create the Python Environment

The project uses Python 3.11.

Load Python on Stanage:

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0
```

If that exact module is no longer available, check the currently installed Python modules:

```bash
module avail Python
```

and load an available Python 3.11 module.

Create a project-specific virtual environment in `parscratch`:

```bash
mkdir -p "/mnt/parscratch/users/$USER/venvs"

python -m venv --system-site-packages \
    "/mnt/parscratch/users/$USER/venvs/dissertation"
```

Activate it:

```bash
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"
```

Check the interpreter:

```bash
which python
python --version
```

The expected Python version is 3.11.

---

## 3. Install Dependencies

Install the dependencies from the main experiment directory:

```bash
cd "$DISSERTATION_ROOT"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The environment should be activated whenever the pipeline is run:

```bash
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"
```

---

## 4. Configure the Hugging Face Cache

The Qwen model files are large and should be stored in Stanage `parscratch` rather than the home directory.

```bash
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"
mkdir -p "$HF_HOME"
```

For a new login session, restore the main environment variables with:

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0

source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"

export REPO_ROOT="$HOME/DissertationCode-FIXED-"
export DISSERTATION_ROOT="$REPO_ROOT/Phase01HPC/ThesisWork"
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"
```

---

## 5. Prepare the LegalBench Data

Move to the preparation directory:

```bash
cd "$REPO_ROOT/preparations"
```

Run the preparation scripts in this order:

```bash
python DowloadingALLLegalBench.py
python ExtractingCandidateTasks.py
python SelectingClassificationTasks.py
python thesis_test.py
```

The final command produces:

```text
preparations/thesisSelection.csv
```

The five tasks used by the main experiment are:

```text
unfair_tos
learned_hands_education
oral_argument_question_purpose
abercrombie
citation_prediction_classification
```

The corresponding task data must be available under:

```text
Phase01HPC/ThesisWork/data/legalbench_csv/
```

with the structure:

```text
legalbench_csv/
├── unfair_tos/
│   ├── train.csv
│   └── test.csv
├── learned_hands_education/
│   ├── train.csv
│   └── test.csv
├── oral_argument_question_purpose/
│   ├── train.csv
│   └── test.csv
├── abercrombie/
│   ├── train.csv
│   └── test.csv
└── citation_prediction_classification/
    ├── train.csv
    └── test.csv
```

If regenerating the data from scratch, copy the selected task data into the main experiment directory:

```bash
mkdir -p "$DISSERTATION_ROOT/data/legalbench_csv"

for TASK in \
    unfair_tos \
    learned_hands_education \
    oral_argument_question_purpose \
    abercrombie \
    citation_prediction_classification
do
    mkdir -p "$DISSERTATION_ROOT/data/legalbench_csv/$TASK"

    cp "$REPO_ROOT/preparations/data/legalbench_all/$TASK/train.csv" \
       "$DISSERTATION_ROOT/data/legalbench_csv/$TASK/train.csv"

    cp "$REPO_ROOT/preparations/data/legalbench_all/$TASK/test.csv" \
       "$DISSERTATION_ROOT/data/legalbench_csv/$TASK/test.csv"
done
```

The following task-specific configuration files are already included in the repository and should remain in:

```text
Phase01HPC/ThesisWork/data/task_field_map.json
Phase01HPC/ThesisWork/data/question_templates.json
```

---

## 6. Build Demonstrations and Evaluation Pools

The demonstration sets and evaluation pools are built using the supplied SLURM job.

Before submitting it, check:

```text
Phase01HPC/ThesisWork/slurm/build_demos.sbatch
```

for account-specific Python or repository paths and replace them with paths for your own Stanage account.

Then submit the job:

```bash
cd "$DISSERTATION_ROOT/slurm"

sbatch build_demos.sbatch
```

Check its status with:

```bash
squeue -u "$USER"
```

The job runs the following scripts in order:

```text
scripts/build_demo_n_shot.py
scripts/build_eval_pools.py
```

and generates:

```text
Phase01HPC/ThesisWork/demonstrations/
Phase01HPC/ThesisWork/eval_pools/
```

Wait for this job to complete before submitting the model-generation jobs.

---

## 7. Cache the Qwen Models

The generation jobs use:

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
```

The models should be downloaded before submitting the GPU batch jobs.

From a Stanage login node:

```bash
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"

export DISSERTATION_ROOT="$HOME/DissertationCode-FIXED-/Phase01HPC/ThesisWork"
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"

cd "$DISSERTATION_ROOT"
```

Run the three model-download scripts:

```bash
python scripts/download_data_qwen7b.py
python scripts/download_data_qwen14b.py
python scripts/download_data_qwen32b.py
```

Each command should finish by confirming that the corresponding model has been cached successfully.

The model downloads are performed on a Stanage login node before the generation jobs are submitted so that the batch jobs can load the models from the shared Hugging Face cache.

---

## 8. Configure the SLURM Generation Scripts

The generation scripts are located under:

```text
Phase01HPC/ThesisWork/slurm/
```

**Before submitting any `.sbatch` file in this repository, check it for account-specific paths and replace them with paths for your own Stanage account.**

The supplied generation scripts contain paths from the account on which the original experiments were run.

The Python interpreter should point to the virtual environment created above:

```bash
/mnt/parscratch/users/$USER/venvs/dissertation/bin/python
```

The project root should point to the cloned repository:

```bash
$HOME/DissertationCode-FIXED-/Phase01HPC/ThesisWork
```

For example, replace the corresponding environment section of each generation script with:

```bash
PYTHON="/mnt/parscratch/users/$USER/venvs/dissertation/bin/python"

export DISSERTATION_ROOT="$HOME/DissertationCode-FIXED-/Phase01HPC/ThesisWork"
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"

cd "$DISSERTATION_ROOT"

$PYTHON --version
```

The generation jobs use the following resources:

| Model | GPUs | Memory | Time limit |
| --- | ---: | ---: | ---: |
| Qwen2.5-7B-Instruct | 1 × A100 | 64 GB | 4 days |
| Qwen2.5-14B-Instruct | 1 × A100 | 64 GB | 4 days |
| Qwen2.5-32B-Instruct | 2 × A100 | 128 GB | 4 days |

The underlying generation commands are:

### Qwen2.5-7B-Instruct

```bash
python scripts/results_generation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --sample_size 45
```

### Qwen2.5-14B-Instruct

```bash
python scripts/results_generation.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --sample_size 45
```

### Qwen2.5-32B-Instruct

```bash
python -u scripts/results_generation.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --sample_size 45
```

These commands should be executed through the corresponding SLURM batch scripts rather than directly on a login node.

---

## 9. Submit the Generation Jobs

Move to the SLURM directory:

```bash
cd "$DISSERTATION_ROOT/slurm"
```

Submit the three generation jobs:

```bash
sbatch run_generations_qwen7B.sbatch
sbatch run_generations_qwen14B.sbatch
sbatch run_generations_qwen32B.sbatch
```

The three jobs are independent and can be submitted concurrently.

Check their status with:

```bash
squeue -u "$USER"
```

To cancel a job:

```bash
scancel <job_id>
```

SLURM writes standard output and error logs to the `.out` and `.err` files specified by each batch script.

Raw generations are written under:

```text
Phase01HPC/ThesisWork/outputs/raw_generations/
```

### Resuming an Interrupted Generation Job

`results_generation.py` supports checkpoint/resume behaviour.

If a generation job is interrupted or reaches its SLURM time limit, submit the same job again:

```bash
sbatch <same-generation-script>.sbatch
```

Existing completed generation records are detected and skipped, allowing generation to continue without repeating completed combinations.

---

## 10. Run the Evaluation Pipeline

Run the evaluation pipeline after the generation jobs have completed.

Move to the evaluation directory:

```bash
cd "$REPO_ROOT/Evaluation"
```

Set the experiment root:

```bash
export DISSERTATION_ROOT="$REPO_ROOT/Phase01HPC/ThesisWork"
```

Run the evaluation pipeline for all three models:

```bash
for MODEL in \
    "Qwen/Qwen2.5-7B-Instruct" \
    "Qwen/Qwen2.5-14B-Instruct" \
    "Qwen/Qwen2.5-32B-Instruct"
do
    echo "=================================================="
    echo "Running evaluation pipeline for $MODEL"
    echo "=================================================="

    python parse_predictions.py --model "$MODEL"
    python diagnose_parsing_failures.py --model "$MODEL"
    python compute_metrics.py --model "$MODEL"
    python analysis.py --model "$MODEL"
    python visualize_metrics.py --model "$MODEL"

    echo "Done with $MODEL"
done
```

The generated outputs are written to:

```text
Phase01HPC/ThesisWork/outputs/parsed_predictions/
Phase01HPC/ThesisWork/results/
Phase01HPC/ThesisWork/figures/
```

---

## 11. Run Cross-Model and Final Analyses

After the per-model evaluation pipeline has completed for all three models:

```bash
cd "$REPO_ROOT/Evaluation"
```

Run the cross-model significance analysis:

```bash
python cross_model_significance.py
```

Then run the final pooled and cross-model analysis scripts:

```bash
python make_mcnemar_champions_filtered.py
python make_dispersion_vs_penalty.py
python make_generic_vs_framework_chart.py
python make_pooled_variance_decomposition.py
```

These scripts write their outputs to the main `results/` and `figures/` directories.

---

## 12. Output Locations

After a complete run, the main generated artefacts are located under:

```text
Phase01HPC/ThesisWork/
│
├── demonstrations/
├── eval_pools/
│
├── outputs/
│   ├── raw_generations/
│   └── parsed_predictions/
│
├── results/
│
└── figures/
```

---

## Optional: Validation Before the Full Generation Sweep

Validation code is available under:

```text
Phase00Testing/
```

Before submitting a testing `.sbatch` file, check it for account-specific paths and replace them where necessary.

A small end-to-end smoke test can be submitted with:

```bash
cd "$REPO_ROOT/Phase00Testing"

sbatch slurm/smoke_test.sbatch
```

The validation code is optional and is not required to run the main reproduction pipeline.

---

## Quick Execution Order

The complete execution order is:

```text
1. Clone the repository
2. Create and activate the Python environment
3. Install requirements
4. Configure DISSERTATION_ROOT and HF_HOME
5. Run the LegalBench preparation scripts
6. Copy the selected tasks into data/legalbench_csv/
7. Submit build_demos.sbatch
8. Wait for demonstrations and evaluation pools to be generated
9. Cache Qwen2.5-7B, Qwen2.5-14B and Qwen2.5-32B
10. Edit the account-specific paths in the generation SLURM scripts
11. Submit the three generation jobs
12. Wait for generation to complete
13. Parse predictions
14. Diagnose parsing failures
15. Compute metrics
16. Run per-model analysis
17. Generate per-model visualisations
18. Run cross-model significance analysis
19. Run the final pooled/cross-model analysis scripts
```