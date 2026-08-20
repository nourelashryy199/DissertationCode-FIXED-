

import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE00_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(PHASE00_DIR)
PHASE1_DIR = os.path.join(REPO_ROOT, "Phase01HPC", "ThesisWork")

if not os.path.isdir(PHASE1_DIR):
    print(f"ERROR: expected to find ThesisWork at {PHASE1_DIR} — adjust the path if your repo layout differs.")
    sys.exit(1)

sys.path.insert(0, PHASE1_DIR)

import config
import schema
import strategy_functions as sf

FAILURES = []


def check(condition, description):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    if not condition:
        FAILURES.append(description)


# ------------------------------------------------------------
# 1. Schema construction
# ------------------------------------------------------------
print("\n=== 1. Schema objects ===")

dummy_task = schema.LegalTask(
    task_id="dummy_0",
    task_type="rule-application_conclusion",  # matches the current merged category name
    context="The mark 'Ice' for an ice cream shop.",
    question="How should this trademark be classified?",
    label_options=["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
    expected_output="descriptive",
)
check(dummy_task.task_id == "dummy_0", "LegalTask constructs with expected fields")
check(dummy_task.task_type in config.CATEGORIES, "dummy_task.task_type matches a real current category name")

demo_1 = schema.Demonstration(context="The mark 'Pictures' for a photography service.", question="", label="generic")
demo_2 = schema.Demonstration(context="The mark 'Shark' for a custom t-shirt maker.", question="", label="arbitrary")
demo_3 = schema.Demonstration(context="The mark 'Kodak' for a camera company.", question="", label="fanciful")

check("Final Answer: generic" in demo_1.render(), "Demonstration.render() includes Final Answer line")


# ------------------------------------------------------------
# 2. Prompt construction — all 13 strategies must build without error
# ------------------------------------------------------------
print("\n=== 2. Strategy prompt construction ===")

# demonstration_sets is now nested per-k: {task_id: {k: [Demonstration, ...]}}.
# Each k has a GENUINELY DIFFERENT demo set here — this is the corrected
# behavior (previously few_shot_2 and few_shot_3 were forced to share the
# same 2 demonstrations; that bug is now fixed, and this test verifies it).
demonstration_sets = {
    "dummy_0": {
        1: [demo_1],
        2: [demo_1, demo_2],
        3: [demo_1, demo_2, demo_3],
    }
}

built_prompts = {}
for strategy in config.ALL_STRATEGIES:
    try:
        prompt = sf.build_prompt(dummy_task, strategy, rephrasing_id=0, task_id="dummy_0",
                                  demonstration_sets=demonstration_sets)
        built_prompts[strategy] = prompt
        ok = isinstance(prompt, str) and len(prompt) > 0 and config.FINAL_ANSWER_PREFIX in prompt
        check(ok, f"'{strategy}' builds a non-empty prompt containing the Final Answer instruction")
    except Exception as e:
        check(False, f"'{strategy}' builds without raising an exception (raised: {e})")

if "few_shot_2" in built_prompts and "few_shot_3" in built_prompts:
    # CORRECTED expectation: with genuinely independent k=2 and k=3
    # demonstration sets available, few_shot_2 and few_shot_3 must now
    # produce DIFFERENT prompts (previously this test asserted they were
    # IDENTICAL, which encoded the old bug as expected behavior).
    check(
        built_prompts["few_shot_2"] != built_prompts["few_shot_3"],
        "few_shot_2 and few_shot_3 now produce DIFFERENT prompts when independent "
        "k=2 and k=3 demonstration sets are available (confirms the fix — "
        "previously these were forced identical; see Methodology for the original limitation)"
    )
    check(
        demo_3.render() in built_prompts["few_shot_3"] and demo_3.render() not in built_prompts["few_shot_2"],
        "the third demonstration appears in few_shot_3's prompt but NOT in few_shot_2's — "
        "confirms each strategy pulls its own exact, independently-sized demo set"
    )

if "one_shot" in built_prompts:
    check(
        demo_1.render() in built_prompts["one_shot"] and demo_2.render() not in built_prompts["one_shot"],
        "one_shot's prompt contains exactly its single k=1 demonstration, not the k=2/k=3 ones"
    )

zero_shot_variants = {
    sf.build_prompt(dummy_task, "zero_shot", r, "dummy_0", demonstration_sets)
    for r in range(config.N_REPHRASINGS)
}
check(len(zero_shot_variants) == config.N_REPHRASINGS,
      f"zero_shot produces {config.N_REPHRASINGS} distinct rephrasing variants")

for framework, steps in config.FRAMEWORK_STEPS.items():
    prompt = built_prompts.get(framework, "")
    all_steps_present = all(step in prompt for step in steps)
    check(all_steps_present, f"'{framework}' prompt mentions all its canonical steps {steps}")


# ------------------------------------------------------------
# 3. Answer parsing (config.extract_final_answer)
# ------------------------------------------------------------
print("\n=== 3. Answer parsing edge cases ===")

check(config.extract_final_answer("blah blah\nFinal Answer: Yes") == "Yes",
      "extracts a simple, well-formed final answer")

check(config.extract_final_answer("no final answer line here") is None,
      "returns None when no Final Answer line is present")

multi = "Final Answer: Draft\nsome more reasoning\nFinal Answer: Confirmed"
check(config.extract_final_answer(multi) == "Confirmed",
      "takes the LAST occurrence when Final Answer appears multiple times (framework restatement case)")

check(config.extract_final_answer("Final Answer:   Yes   ") == "Yes",
      "strips surrounding whitespace from the extracted answer")

check(config.extract_final_answer("Final Answer: Yes.") == "Yes.",
      "extraction is exact-match (trailing period NOT stripped here — normalization now happens "
      "inline in results_generation.py's normalize_answer(), not a separate rescore.py step)")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)