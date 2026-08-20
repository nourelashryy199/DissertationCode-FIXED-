import re #regular expressions, for extracting final answers from model output (raw text)
import os #file path handling and reading environment variables
import argparse #parses command-line arguments (for model name and sample size)

#model settings
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" #which model to use if nothing is specified on the command line.
#this is a fallback default; and the actual model name is determined by the --model CLI arg passed to run_stage_a.py, which is parsed by get_model_name_from_args() below.
MAX_NEW_TOKENS = 512 #token budget for each generation

#sampling/rephrasing design
TEMPERATURE = 0.7 
TOP_P = 0.95
#both temperature and top_p are used to control the randomness of the model's word choices during generation.
#higher temperature means more randomness, and lower temperature means more deterministic output.
#higher top_p means the model considers a larger set of possible next words, while lower top_p means it focuses on the most likely next words.
N_REPHRASINGS = 3
#each strategy gets 3 differently-worded but equivalent instructions.
N_RUNS = 3
#each strategy is run 5 times per task, to account for randomness in the model's output (capture sampling variance).

# #evaluation subsampling
# EVAL_SAMPLE_SIZE = 45 
# #a single shared constant for "how many instances per task to evaluate"

#LegalBench reasoning-type categories
CATEGORIES = [
    "issue-spotting",
    "rule-recall",
    "rule-application_conclusion",
    "interpretation",
    "rhetorical-understanding",
]

#prompting strategies
GENERIC_STRATEGIES = [
    "zero_shot",
    "one_shot",
    "few_shot_2",
    "few_shot_3",
    "role_based",
    "structured",
    "cot",
]

LEGAL_FRAMEWORK_STRATEGIES = [
    "irac",
    "crac",
    "creac",
    "cleo",
    "treacc",
    "ireac",
]

ALL_STRATEGIES = GENERIC_STRATEGIES + LEGAL_FRAMEWORK_STRATEGIES

DEMO_REQUIRED_STRATEGIES = {
    "one_shot": 1,
    "few_shot_2": 2,
    "few_shot_3": 3,
}

# Canonical legal reasoning framework step sequences
# Sourced from Burton (2017) / Turner (2012)
# for full citation and discussion of acronym instability
# (e.g. "IRREAC" in the AI-prompting literature vs. the
# pedagogically-standard IREAC used here).
FRAMEWORK_STEPS = {
    "irac":   ["Issue", "Rule", "Application", "Conclusion"],
    "crac":   ["Conclusion", "Rule", "Application", "Conclusion"],
    "creac":  ["Conclusion", "Rule", "Explanation", "Application", "Conclusion"],
    "cleo":   ["Claim", "Law", "Evaluation", "Outcome"],
    "treacc": ["Topic", "Rule", "Explanation", "Analysis", "Counterarguments", "Conclusion"],
    "ireac":  ["Issue", "Rule", "Explanation", "Application", "Conclusion"],
}

# Parsing convention 
FINAL_ANSWER_PREFIX = "Final Answer:" #the literal string that the model is instructed to use to indicate its final answer in its output
FINAL_ANSWER_INSTRUCTION = (
    f"End your response with a line in exactly this format: {FINAL_ANSWER_PREFIX} <your answer>"
) # the actual sentence appended to every prompt telling the model to do this.


def extract_final_answer(raw_output: str) -> str | None:
    """
    Extracts the model's final answer, taking the LAST occurrence
    of the Final Answer prefix — important because some legal
    reasoning frameworks (e.g., CRAC, CREAC) instruct the model to
    restate a conclusion mid-reasoning, and some models echo the
    instruction text before the true final line.
    """
    matches = re.findall(r"Final Answer:\s*(.+)", raw_output)
    #scans the model's raw output for any lines that start with "Final Answer:" followed by optional whitespace 
    #returns a list of all matches; could be 0, 1 or more matches depending on how many times the model included that line in its output.
    return matches[-1].strip() if matches else None
#takes last match (if any) and trims leading/trailing whitespaces. 
#if the list is empty, this is exactly what counts as a parsing failure in parse_predictions.py


# Clustering / demonstration selection
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" #which sentence embedding model to use for clustering and demonstration selection
CLUSTERING_RANDOM_STATE = 42 #a fixed random seed for reproducibility of clustering results (e.g., KMeans) across runs.

# Paths
# HPC_ROOT should be set to wherever this repo is cloned on Stanage
# (e.g. /users/msp25noe/dissertation/phase1_hpc), passed via the
# DISSERTATION_ROOT environment variable set in each .sbatch script,
# so paths aren't hardcoded to any one user's home directory layout.
HPC_ROOT = os.environ.get("DISSERTATION_ROOT", os.getcwd())
#reads the DISSERTATION_ROOT environment variable (if set) to determine the root directory for all data and output paths. If not set, defaults to the current working directory.
DATA_DIR = os.path.join(HPC_ROOT, "data", "legalbench_csv")
EVAL_POOLS_DIR = os.path.join(HPC_ROOT, "data", "eval_pools")
MANIFEST_PATH = os.path.join(HPC_ROOT, "data", "manifest.csv")
DEMO_DIR = os.path.join(HPC_ROOT, "demonstrations")
RAW_GEN_DIR = os.path.join(HPC_ROOT, "outputs", "raw_generations")
PARSED_DIR = os.path.join(HPC_ROOT, "outputs", "parsed_predictions")
LOG_DIR = os.path.join(HPC_ROOT, "outputs", "logs")
RESULTS_DIR = os.path.join(HPC_ROOT, "results")
FIGURES_DIR = os.path.join(HPC_ROOT, "figures")

#CLI (command-line interface) argument parsing for model name and sample size 
def get_model_name_from_args() -> str:
    """
    Allows each SLURM job to specify which model to run via
    `python run_stage_a.py --model meta-llama/Llama-3.3-70B-Instruct`,
    so one script serves the entire Qwen -> Llama progression without
    editing this file per job.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sample_size", type=int, default=None,
                         help="Optional: limit instances per task (for testing). "
                              "Omit for full Phase 1 eval pools.")
    args, _ = parser.parse_known_args()
    return args
#creates an argument parser with two options:
# -- model: defalts to 7B model if not given, 
# -- sample_size: defaults to None if not given, which means the full evaluation pool will be used.
#parse_known_args() is used instead of parse_args() to allow for additional arguments to be passed in without causing an error, which is useful in a SLURM job context where other arguments may be present.
#returns the parsed args object itself (not just the model name, despite the function's name) — so callers access args.model and args.sample_size from it, as seen in run_stage_a.py's main().