# Consistency Matters: A Risk-Adjusted Approach to Evaluating Prompt Strategies for Legal Reasoning

Code and experimental pipeline for the MSc Advanced Computer Science dissertation:

**“Consistency Matters: A Risk-Adjusted Approach to Evaluating Prompt Strategies for Legal Reasoning”**

University of Sheffield, 2026.

This project investigates whether reliable prompting strategies vary across categories of legal reasoning and model scales. Rather than selecting prompts using mean accuracy alone, the study evaluates improvement over a zero-shot reference while accounting for variation across repeated stochastic generation runs and instruction rephrasings. The evaluation includes Run, Rephrasing, and Joint Fit Scores, Transfer Penalty, Strategy Dispersion, variance decomposition, and statistical comparisons.

Thirteen prompting strategies are evaluated in total. Zero-shot is the reference baseline for risk-adjusted improvement; the remaining 12 strategies form the candidate set used for Fit-based champion selection and transfer analysis. The RQ4 generic-versus-framework comparison intentionally includes zero-shot among the seven generic strategies.

## Quick Start

The full generation sweep was run on the University of Sheffield **Stanage** HPC cluster using NVIDIA A100 GPUs. A reviewer can perform the lightweight preparation and evaluation stages in any compatible Python environment, but the supplied SLURM workflow is written for Stanage.

The complete reproduction sequence is:

```text
1. Connect to Stanage and clone the repository
2. Create a Python 3.11 environment and install requirements.txt
3. Set REPO_ROOT, DISSERTATION_ROOT, and HF_HOME
4. Run the LegalBench preparation pipeline
5. Copy the five selected tasks into ThesisWork/data/legalbench_csv/
6. Edit PYTHON and DISSERTATION_ROOT in the supplied .sbatch files
7. Build demonstrations and full evaluation pools
8. Cache Qwen2.5-7B, 14B, and 32B
9. Submit the three generation jobs with --sample_size 45
10. Wait for generation to complete (resubmit interrupted jobs if necessary)
11. Parse predictions and run the per-model evaluation pipeline
12. Run cross-model and pooled analyses
13. Inspect results/ and figures/
```

The intended generation design contains:

```text
45 evaluation instances
× 13 strategies
× 3 instruction rephrasings
× 3 stochastic runs
= 5,265 generations per task/model

5 tasks × 5,265 = 26,325 generations per model

3 models × 26,325 = 78,975 intended generation conditions
```

The evaluation sample is fixed by seed 42, but stochastic decoding does **not** use an explicit generation seed. The procedure can therefore be reproduced, but individual generated responses and resulting aggregate values are not guaranteed to be byte-for-byte identical across reruns.

---

## Experimental Setup

### LegalBench tasks

One classification task represents each of five LegalBench reasoning categories:

| Legal reasoning category | Task |
|---|---|
| Interpretation | `unfair_tos` |
| Issue-Spotting | `learned_hands_education` |
| Rhetorical-Understanding | `oral_argument_question_purpose` |
| Rule-Application/Conclusion | `abercrombie` |
| Rule-Recall | `citation_prediction_classification` |

Each model is evaluated on a deterministic fixed-seed sample of 45 test instances from each task.

### Prompting strategies

Seven generic strategies are evaluated:

- `zero_shot` — reference baseline
- `one_shot`
- `few_shot_2`
- `few_shot_3`
- `role_based`
- `structured`
- `cot`

Six legal reasoning frameworks are evaluated:

- `irac`
- `crac`
- `creac`
- `cleo`
- `treacc`
- `ireac`

There are therefore **13 evaluated strategies**. For Fit-based candidate ranking, champion selection, Strategy Dispersion, and transfer analysis, zero-shot supplies the reference baseline and the other 12 strategies form the candidate set.

### Models and repeated conditions

The experiment uses:

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
```

Each strategy is evaluated using three instruction rephrasings and three stochastic generation runs. Generation uses sampling with `temperature=0.7`, `top_p=0.95`, and `max_new_tokens=512`.

---

## Repository Structure

The main repository components are:

```text
DissertationCode-FIXED-/
│
├── preparations/
│   ├── DowloadingALLLegalBench.py
│   ├── ExtractingCandidateTasks.py
│   ├── SelectingClassificationTasks.py
│   ├── thesis_test.py
│   ├── candidate_tasks.csv
│   ├── finalSelection.csv
│   ├── thesisSelection.csv
│   └── data/
│       └── legalbench_all/
│
├── Phase01HPC/
│   └── ThesisWork/
│       ├── config.py
│       ├── schema.py
│       ├── model.py
│       ├── strategy_functions.py
│       ├── requirements.txt
│       │
│       ├── data/
│       │   ├── legalbench_csv/
│       │   ├── task_field_map.json
│       │   └── question_templates.json
│       │
│       ├── scripts/
│       │   ├── build_demo_n_shot.py
│       │   ├── build_eval_pools.py
│       │   ├── download_data_qwen7b.py
│       │   ├── download_data_qwen14b.py
│       │   ├── download_data_qwen32b.py
│       │   └── results_generation.py
│       │
│       ├── slurm/
│       │   ├── build_demos.sbatch
│       │   ├── run_generations_qwen7B.sbatch
│       │   ├── run_generations_qwen14B.sbatch
│       │   ├── run_generations_qwen32B.sbatch
│       │   └── test_gpu.sbatch
│       │
│       ├── demonstrations/
│       ├── eval_pools/
│       ├── outputs/
│       │   ├── raw_generations/
│       │   └── parsed_predictions/
│       ├── results/
│       └── figures/
│
└── Evaluation/
    ├── analysis.py
    ├── compute_metrics.py
    ├── cross_model_significance.py
    ├── diagnose_parsing_failures.py
    ├── make_dispersion_vs_penalty.py
    ├── make_generic_vs_framework_chart.py
    ├── make_mcnemar_champions_filtered.py
    ├── make_pooled_variance_decomposition_chart.py
    ├── make_pooled_variance_decomposition.py
    ├── parse_predictions.py
    └── visualize_metrics.py
```

`preparations/` implements LegalBench download and task selection. `Phase01HPC/ThesisWork/` contains the main experiment. `Evaluation/` parses generations, computes metrics, performs the statistical analyses, and generates figures.

---

# Reproducing the Experiments

## 1. Connect to Stanage

Connect to Stanage using the access procedure provided by the University of Sheffield. The commands below begin **after a Stanage login has been established**.

The supplied SLURM files were used on Stanage and request NVIDIA A100 GPUs. They contain paths from the account on which the original experiment was run, so two lines must be changed before another user submits them. This is covered in Step 7.

---

## 2. Clone the Repository

From a Stanage login node:

```bash
cd "$HOME"
git clone https://github.com/nourelashryy199/DissertationCode-FIXED-.git
cd DissertationCode-FIXED-
```

Define reusable paths:

```bash
export REPO_ROOT="$HOME/DissertationCode-FIXED-"
export DISSERTATION_ROOT="$REPO_ROOT/Phase01HPC/ThesisWork"
```

The original experimenter's checkout happened to be under:

```text
/users/msp25noe/Thesis_Code/DissertationCode-FIXED-/
```

That historical path is **not required**. A fresh clone may be placed elsewhere as long as `DISSERTATION_ROOT` and the SLURM files point to the actual location.

`REPO_ROOT` and `DISSERTATION_ROOT` are shell variables and should be set again in a new login session.

---

## 3. Create the Python Environment

The final experimental environment used Python **3.11.13**. The main verified package versions are pinned in:

```text
Phase01HPC/ThesisWork/requirements.txt
```

The original environment reported PyTorch `2.5.1+cu124`; the requirements file specifies `torch==2.5.1` together with the remaining pinned dependencies.

### 3.1 Load Python 3.11 on Stanage

A Python 3.11 module can be used to create a clean environment:

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0
```

If that exact module is no longer available:

```bash
module avail Python
```

and load an available Python 3.11 module.

### 3.2 Create a virtual environment

A convenient location is Stanage `parscratch`:

```bash
mkdir -p "/mnt/parscratch/users/$USER/venvs"

python -m venv --system-site-packages \
    "/mnt/parscratch/users/$USER/venvs/dissertation"

source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"
```

Verify the interpreter:

```bash
which python
python --version
```

Do **not** run the project with Stanage's default `/usr/bin/python`; on the original login node this resolved to Python 2.7.5 before the project environment was selected.

### 3.3 Install dependencies

```bash
cd "$DISSERTATION_ROOT"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned requirements are:

```text
torch==2.5.1
transformers==4.46.3
accelerate==1.1.1
sentence-transformers==3.2.1
scikit-learn==1.5.2
scipy==1.13.1
pandas==2.2.3
numpy==1.26.4
pyarrow==14.0.2
datasets==2.19.0
Pillow==10.4.0
matplotlib==3.7.5
contourpy==1.1.1
seaborn==0.12.2
```

The final experimental environment was verified with these core versions. A compatible CUDA-enabled PyTorch installation is required for GPU generation on Stanage.

---

## 4. Configure the Hugging Face Cache

The Qwen checkpoints are large. Store the Hugging Face cache in `parscratch` rather than the home directory:

```bash
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"
mkdir -p "$HF_HOME"
```

A convenient setup block for a new Stanage login is therefore:

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"

export REPO_ROOT="$HOME/DissertationCode-FIXED-"
export DISSERTATION_ROOT="$REPO_ROOT/Phase01HPC/ThesisWork"
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"
```

Adjust the module or repository location if your installation differs.

---

## 5. Prepare the LegalBench Data

The preparation pipeline reproduces the task-selection process used in the dissertation.

Move to:

```bash
cd "$REPO_ROOT/preparations"
```

### 5.1 Download LegalBench

```bash
python DowloadingALLLegalBench.py
```

This exports the available LegalBench task splits as CSV files under:

```text
preparations/data/legalbench_all/
```

### 5.2 Identify candidate classification tasks

```bash
python ExtractingCandidateTasks.py
```

The classification screening uses:

```text
number of unique labels <= 20
unique-label ratio <= 0.1
```

where the unique-label ratio is the number of unique labels divided by the number of test instances. This is a screening criterion, not a class-balance measure.

The script produces:

```text
preparations/candidate_tasks.csv
```

### 5.3 Select tasks by legal-reasoning category

```bash
python SelectingClassificationTasks.py
```

Rule-Application and Rule-Conclusion are combined into the `rule-application_conclusion` category for this experiment. Candidate tasks are ranked within category by number of unique labels and then unique-label ratio, and the selection stage produces:

```text
preparations/finalSelection.csv
```

### 5.4 Produce the final five-task thesis selection

```bash
python thesis_test.py
```

Despite its filename, `thesis_test.py` is part of the task-selection pipeline rather than a unit test. It produces:

```text
preparations/thesisSelection.csv
```

The final tasks are:

| Category | Task |
|---|---|
| Interpretation | `unfair_tos` |
| Issue-Spotting | `learned_hands_education` |
| Rhetorical-Understanding | `oral_argument_question_purpose` |
| Rule-Application/Conclusion | `abercrombie` |
| Rule-Recall | `citation_prediction_classification` |

### 5.5 Copy the selected task data into the main experiment

The preparation directory and main experiment use separate data locations. Copy each selected task's `train.csv` and `test.csv`:

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

The final structure should be:

```text
Phase01HPC/ThesisWork/data/legalbench_csv/
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

The repository already contains the task-specific configuration files:

```text
Phase01HPC/ThesisWork/data/task_field_map.json
Phase01HPC/ThesisWork/data/question_templates.json
```

---

## 6. Build Demonstrations and Evaluation Pools

Two scripts prepare the inputs consumed by `results_generation.py`:

```text
scripts/build_demo_n_shot.py
scripts/build_eval_pools.py
```

The supplied `slurm/build_demos.sbatch` runs both scripts sequentially. They can also be run manually.

### 6.1 Demonstration selection

Demonstrations are selected from each task's **training split only** using `sentence-transformers/all-MiniLM-L6-v2`.

Selection is performed independently for each requested value of `k`:

- `k=1`: the training instance nearest the mean embedding is selected.
- `k=2` and `k=3`: K-means is run independently with `random_state=42` and `n_init=10`; instances nearest the cluster centroids are selected.
- If clustering does not supply enough distinct representatives, the implemented fallback fills the remaining positions with unused examples selected for diversity.

The resulting JSON files are written under:

```text
Phase01HPC/ThesisWork/demonstrations/
```

### 6.2 Expected demonstration behaviour

For four tasks, the script produces `k=1`, `k=2`, and `k=3` files.

`citation_prediction_classification` contains only two training instances, so its expected outputs are only:

```text
citation_prediction_classification_demos_k1.json
citation_prediction_classification_demos_k2.json
```

The warning that `train_pool=2 < k=3` is expected.

During the verified run, `oral_argument_question_purpose` produced scikit-learn `ConvergenceWarning`s for `k=2` and `k=3` because clustering returned fewer distinct clusters than requested. The implemented fallback then filled the remaining demonstration positions. These warnings were non-fatal and demonstration construction completed.

A successful run produces **14 demonstration JSON files** in total.

### 6.3 Evaluation pools

`build_eval_pools.py` writes the **complete test split** for each selected task to:

```text
Phase01HPC/ThesisWork/eval_pools/
```

The verified pool sizes were:

| Task | Full evaluation-pool size |
|---|---:|
| `unfair_tos` | 3,813 |
| `learned_hands_education` | 56 |
| `oral_argument_question_purpose` | 312 |
| `abercrombie` | 95 |
| `citation_prediction_classification` | 108 |

A successful run produces **5 evaluation-pool JSON files**.

These are not yet the final 45-instance experimental samples. `results_generation.py` later applies the 45-instance restriction.

### 6.4 Recommended Stanage route: submit the supplied batch job

Before submission, edit `slurm/build_demos.sbatch` as described in Step 7, because the supplied file contains the original user's hard-coded Python and repository paths.

Then:

```bash
cd "$DISSERTATION_ROOT"
sbatch slurm/build_demos.sbatch
```

Monitor it with:

```bash
squeue -u "$USER"
```

Wait for it to finish before starting generation.

### 6.5 Equivalent manual route

With the project environment active:

```bash
cd "$DISSERTATION_ROOT"

python scripts/build_demo_n_shot.py
python scripts/build_eval_pools.py
```

This is equivalent to the two Python commands executed by `build_demos.sbatch`.

---

## 7. Edit the Supplied SLURM Files for Your Stanage Account

**Do this before submitting `build_demos.sbatch` or any generation `.sbatch` file.**

The supplied batch files contain the original experimenter's absolute Python and repository paths. Another user must replace **two values**.

Files to edit:

```text
Phase01HPC/ThesisWork/slurm/build_demos.sbatch
Phase01HPC/ThesisWork/slurm/run_generations_qwen7B.sbatch
Phase01HPC/ThesisWork/slurm/run_generations_qwen14B.sbatch
Phase01HPC/ThesisWork/slurm/run_generations_qwen32B.sbatch
```

### 7.1 Replace `PYTHON`

The supplied files contain an original-account path similar to:

```bash
PYTHON=/users/msp25noe/.conda/envs/dissertation_env/bin/python3
```

If you created the virtual environment described above, replace it with:

```bash
PYTHON="/mnt/parscratch/users/$USER/venvs/dissertation/bin/python"
```

### 7.2 Replace `DISSERTATION_ROOT`

The supplied files contain:

```bash
export DISSERTATION_ROOT=/users/msp25noe/Thesis_Code/DissertationCode-FIXED-/Phase01HPC/ThesisWork
```

For the default clone location in this README, replace it with:

```bash
export DISSERTATION_ROOT="$HOME/DissertationCode-FIXED-/Phase01HPC/ThesisWork"
```

### 7.3 `HF_HOME` does not need a username edit

The supplied line is already portable:

```bash
export HF_HOME=/mnt/parscratch/users/$USER/hf_cache
```

### 7.4 Verify the edited files

Before submission:

```bash
grep -nE 'PYTHON=|DISSERTATION_ROOT=|HF_HOME=' \
    slurm/build_demos.sbatch \
    slurm/run_generations_qwen7B.sbatch \
    slurm/run_generations_qwen14B.sbatch \
    slurm/run_generations_qwen32B.sbatch
```

There should be no remaining `/users/msp25noe/...` path in the active configuration lines.

---

## 8. Cache the Qwen Models

The experiment uses:

```text
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-14B-Instruct
Qwen/Qwen2.5-32B-Instruct
```

Set the cache and move to the main experiment:

```bash
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"
export HF_HOME="/mnt/parscratch/users/$USER/hf_cache"
cd "$DISSERTATION_ROOT"
```

Run:

```bash
python scripts/download_data_qwen7b.py
python scripts/download_data_qwen14b.py
python scripts/download_data_qwen32b.py
```

The purpose of this stage is to populate the shared Hugging Face cache before the GPU generation jobs load the models.

If Hugging Face authentication is required by the environment at the time of reproduction, authenticate using the standard Hugging Face tooling before running these scripts.

---

## 9. Run the Generation Sweep

### 9.1 Resource requests

The supplied generation jobs request:

| Model | A100 GPUs | System memory | Time limit |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | 1 | 64 GB | 4 days |
| Qwen2.5-14B-Instruct | 1 | 64 GB | 4 days |
| Qwen2.5-32B-Instruct | 2 | 128 GB | 4 days |

`build_demos.sbatch` requests one A100, 32 GB system memory, and 30 minutes.

These are the resources requested by the supplied Stanage batch files; they should not be interpreted as measurements of minimum hardware requirements.

### 9.2 Underlying commands

The 7B job runs:

```bash
python scripts/results_generation.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --sample_size 45
```

The 14B job runs:

```bash
python scripts/results_generation.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --sample_size 45
```

The 32B job runs:

```bash
python -u scripts/results_generation.py \
    --model Qwen/Qwen2.5-32B-Instruct \
    --sample_size 45
```

Although the CLI help describes `--sample_size` as an optional testing limit, **`--sample_size 45` is part of the dissertation's final experimental protocol and must be supplied to reproduce it.**

### 9.3 How the 45 instances are selected

For each task, `results_generation.py` loads the complete evaluation pool, copies it, shuffles it using Python's `random.Random(42)`, and takes the first 45 instances.

The sample is therefore deterministic for a given pool and seed, but it is **not label-stratified**.

### 9.4 Candidate-label construction

After the 45 evaluation instances are selected, the candidate answer labels supplied to the model are constructed from the gold labels occurring in those selected instances.

This reproduces the experiment as conducted. It is also a documented design limitation: if a task-level label were absent from the selected 45 instances, it would also be absent from the candidate answer options.

### 9.5 Submit the jobs

After checking the edited SLURM paths:

```bash
cd "$DISSERTATION_ROOT"

sbatch slurm/run_generations_qwen7B.sbatch
sbatch slurm/run_generations_qwen14B.sbatch
sbatch slurm/run_generations_qwen32B.sbatch
```

The jobs are independent and may be submitted concurrently, subject to Stanage scheduling and allocation constraints.

Monitor them with:

```bash
squeue -u "$USER"
```

Cancel a job if necessary with:

```bash
scancel <job_id>
```

SLURM writes standard output and error logs using the filenames configured in each batch file.

### 9.6 Raw generation output

Raw generations are written under:

```text
Phase01HPC/ThesisWork/outputs/raw_generations/
```

The experiment evaluates all 13 strategies, 3 rephrasings, and 3 runs for each selected instance.

### 9.7 Checkpoint and resume

`results_generation.py` supports resuming an interrupted run. Existing JSONL output is read on startup; completed generation keys are detected and skipped. Duplicate keys are deduplicated, and malformed partial JSONL lines are skipped when recovering existing progress.

If a job reaches its walltime or is interrupted, resubmit the **same** batch file:

```bash
sbatch slurm/<same-generation-script>.sbatch
```

The job continues from the previously recorded combinations rather than intentionally regenerating completed keys.

### 9.8 Reproducibility of stochastic generation

The following aspects are controlled:

```text
evaluation sampling seed = 42
demonstration clustering seed = 42
three fixed instruction rephrasings
fixed model family/checkpoints
fixed generation hyperparameters
```

Generation itself uses stochastic decoding (`do_sample=True`) with:

```text
temperature = 0.7
top_p = 0.95
max_new_tokens = 512
```

No explicit generation seed is set. Consequently, a fresh reproduction should follow the same experimental design but is not guaranteed to generate identical model responses or identical downstream aggregate values.

---

## 10. Run the Per-Model Evaluation Pipeline

Run evaluation only after the relevant generation output is complete.

Activate the environment and set the experiment root if necessary:

```bash
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"

export REPO_ROOT="$HOME/DissertationCode-FIXED-"
export DISSERTATION_ROOT="$REPO_ROOT/Phase01HPC/ThesisWork"

cd "$REPO_ROOT/Evaluation"
```

Run the pipeline for all three models:

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

The stages perform:

```text
parse_predictions.py
    -> extracts and normalizes final answers

diagnose_parsing_failures.py
    -> summarizes outputs that could not be parsed

compute_metrics.py
    -> computes descriptive performance metrics

analysis.py
    -> computes Fit, champions, transfer, dispersion,
       variance decomposition, correlations, and related analyses

visualize_metrics.py
    -> generates per-model figures
```

Generated artefacts are written primarily under:

```text
Phase01HPC/ThesisWork/outputs/parsed_predictions/
Phase01HPC/ThesisWork/results/
Phase01HPC/ThesisWork/figures/
```

A missing parse is treated as an incorrect prediction in the evaluation pipeline.

---

## 11. Run Cross-Model and Final Analyses

After the per-model pipeline has completed for **all three models**:

```bash
cd "$REPO_ROOT/Evaluation"
```

Run cross-model significance analysis:

```bash
python cross_model_significance.py
```

Then run the final analysis/figure scripts:

```bash
python make_mcnemar_champions_filtered.py
python make_dispersion_vs_penalty.py
python make_generic_vs_framework_chart.py
python make_pooled_variance_decomposition.py
python make_pooled_variance_decomposition_chart.py
```

The last two commands are separate: the first computes the pooled variance-decomposition output and the second generates its chart.

These scripts write their outputs to the main experiment's `results/` and `figures/` directories.

The full cross-model McNemar analysis contains 195 strategy-level tests (13 strategies × 5 categories × 3 model-pair comparisons). Champion-focused outputs are subsequently filtered using the non-zero-shot Joint Fit champions.

---

## 12. Main Output Locations

After a complete run:

```text
Phase01HPC/ThesisWork/
│
├── demonstrations/
│   └── selected one-/few-shot demonstration JSON files
│
├── eval_pools/
│   └── complete task-level test pools
│
├── outputs/
│   ├── raw_generations/
│   │   └── model generation JSONL files
│   └── parsed_predictions/
│       └── parsed model predictions
│
├── results/
│   └── metrics, statistical outputs, Fit/transfer analyses, etc.
│
└── figures/
    └── generated visualisations
```

Useful preprocessing sanity checks are:

```text
14 demonstration JSON files
5 evaluation-pool JSON files
```

Useful intended-generation sanity checks are:

```text
5,265 conditions per task/model
26,325 conditions per model
78,975 conditions across all three models
```

---

## 13. Optional Manual Checks

### Check required task files

From `Phase01HPC/ThesisWork`:

```bash
for TASK in \
    unfair_tos \
    learned_hands_education \
    oral_argument_question_purpose \
    abercrombie \
    citation_prediction_classification
do
    printf "%-42s " "$TASK"

    test -f "data/legalbench_csv/$TASK/train.csv" &&
    test -f "data/legalbench_csv/$TASK/test.csv" &&
    echo "train + test OK" || echo "MISSING"
done
```

### Check configuration files

```bash
for FILE in data/task_field_map.json data/question_templates.json
do
    test -f "$FILE" && echo "$FILE: OK" || echo "$FILE: MISSING"
done
```

### Check generated preprocessing files

```bash
ls -1 demonstrations
ls -1 eval_pools
```

### Check the generation CLI

```bash
python scripts/results_generation.py --help
```

The verified interface accepts:

```text
--model MODEL
--sample_size SAMPLE_SIZE
```

---

## 14. Troubleshooting and Expected Behaviour

### `python` resolves to Python 2.7

On a fresh Stanage login, the system `python` may not be the project interpreter. Load/activate the Python 3.11 environment before running the pipeline:

```bash
source "/mnt/parscratch/users/$USER/venvs/dissertation/bin/activate"
python --version
```

### A batch job refers to `/users/msp25noe/...`

The supplied `.sbatch` files preserve the original experimenter's paths. Replace `PYTHON` and `DISSERTATION_ROOT` as described in Step 7 before submission.

### `citation_prediction_classification` has no 3-shot demonstration

Expected. Its training pool contains only two instances, so the experiment cannot construct three distinct demonstrations.

### `oral_argument_question_purpose` reports a K-means `ConvergenceWarning`

This occurred in the verified preprocessing run. The clustering produced fewer distinct clusters than requested for `k=2` and `k=3`; the script's fallback filled the remaining positions and preprocessing completed successfully.

### Evaluation pools contain more than 45 instances

Expected. `build_eval_pools.py` stores each complete test split. The deterministic 45-instance sample is selected later by `results_generation.py --sample_size 45`.

### A generation job stops after four days

The supplied generation jobs have a four-day walltime. Resubmit the same `.sbatch` file. Checkpoint/resume logic skips already recorded generation combinations.

### A fresh run does not exactly reproduce the original model responses

Expected. Evaluation sampling and clustering are seeded, but stochastic decoding does not use an explicit generation seed. Exact response-level reproduction is therefore not guaranteed.

---

## 15. Reproduction Checklist

After reading the detailed instructions above, the complete workflow is:

```text
[ ] Connect to Stanage
[ ] Clone DissertationCode-FIXED-
[ ] Load Python 3.11
[ ] Create and activate the project environment
[ ] Install requirements.txt
[ ] Set REPO_ROOT, DISSERTATION_ROOT, and HF_HOME
[ ] Download LegalBench
[ ] Run ExtractingCandidateTasks.py
[ ] Run SelectingClassificationTasks.py
[ ] Run thesis_test.py
[ ] Copy the five selected task train/test CSVs into data/legalbench_csv/
[ ] Edit PYTHON and DISSERTATION_ROOT in all four experiment .sbatch files
[ ] Run/submit demonstration and evaluation-pool construction
[ ] Confirm 14 demonstration JSONs and 5 evaluation-pool JSONs
[ ] Cache all three Qwen2.5 checkpoints
[ ] Submit the 7B, 14B, and 32B generation jobs with --sample_size 45
[ ] Monitor/resubmit interrupted jobs until generation is complete
[ ] Run parse_predictions.py for all three models
[ ] Run diagnose_parsing_failures.py for all three models
[ ] Run compute_metrics.py for all three models
[ ] Run analysis.py for all three models
[ ] Run visualize_metrics.py for all three models
[ ] Run cross_model_significance.py
[ ] Run make_mcnemar_champions_filtered.py
[ ] Run make_dispersion_vs_penalty.py
[ ] Run make_generic_vs_framework_chart.py
[ ] Run make_pooled_variance_decomposition.py
[ ] Run make_pooled_variance_decomposition_chart.py
[ ] Inspect outputs/, results/, and figures/
```

---

## Reproducibility Note

The repository preserves the experimental design, task-selection pipeline, prompt strategies, deterministic evaluation sampling procedure, demonstration-selection procedure, generation settings, parsing logic, metrics, statistical analyses, and visualisation code used in the dissertation.

The final experiment used a deterministic seed for evaluation sampling and demonstration clustering, but did not explicitly seed stochastic model decoding. Reproduction should therefore be understood as reproduction of the **experimental procedure and analysis pipeline**, rather than a guarantee of identical generated text on every rerun.
