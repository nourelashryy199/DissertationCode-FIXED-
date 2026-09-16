import config
#imports the shared experiment settings from config.py, including the framework steps, strategy list, demonstration requirements, and final-answer instruction.


#maps each LegalBench reasoning category to the legal domain used in the role-based and structured prompts.
CATEGORY_DOMAIN = {
    "issue-spotting": "general civil litigation",
    "rule-recall": "consumer protection and regulatory law",
    "rule-application_conclusion": "trademark, intellectual property, and contract law",
    "interpretation": "commercial contract interpretation",
    "rhetorical-understanding": "legal reasoning and argumentation",
}

#the 3 differently-worded but equivalent instruction versions used for each applicable prompting strategy.
#rephrasing_id determines which version is used for a particular generation.
INSTRUCTION_REPHRASINGS = {
    "zero_shot": [
        "Classify the following based on the labels provided.",
        "Determine the correct classification for the following, using the given labels.",
        "Read the following and assign the appropriate label from the list provided.",
    ],
    "role_based": [
        "You are an experienced attorney specializing in {domain}. Classify the following based on the labels provided.",
        "As a practicing lawyer with expertise in {domain}, determine the correct classification for the following.",
        "You are a legal professional focused on {domain}. Read the following and assign the appropriate label.",
    ],
    "cot": [
        "Classify the following based on the labels provided. Let's think step by step.",
        "Determine the correct classification for the following. Work through your reasoning step by step before answering.",
        "Read the following and assign the appropriate label. Reason through this carefully, step by step.",
    ],
    "structured": [
        "Using the jurisdiction, practice area, facts, and constraints below, classify the following.",
        "Given the structured case details below, determine the correct classification.",
        "Based on the labeled case information below, assign the appropriate label.",
    ],
}


def build_label_line(task):
    #turns the possible labels for the task into the label line included in the prompt.
    return f"Labels: {', '.join(task.label_options)}"


def build_context_question_block(task):
    #puts the task context and question into the common format used by the prompts.
    return f"Context: {task.context}\nQuestion: {task.question}"


def build_final_answer_instruction():
    #adds the shared instruction telling the model exactly how its final answer should be formatted.
    return config.FINAL_ANSWER_INSTRUCTION


def build_demo_block(demos):
    """Combines the already-selected demonstrations into one block; the correct number of demonstrations has already been selected before this point, so nothing needs to be sliced here."""
    return "\n\n".join(d.render() for d in demos)


def prompt_zero_shot(task, rephrasing_id, demos=None):
    #zero-shot gives the model the instruction, possible labels and task itself, without any demonstrations.
    instruction = INSTRUCTION_REPHRASINGS["zero_shot"][rephrasing_id]
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_n_shot(task, rephrasing_id, demos):
    #common prompt construction used by the one-shot, two-shot and three-shot strategies.
    #the demonstrations passed here have already been independently selected for the required value of k.
    instruction = INSTRUCTION_REPHRASINGS["zero_shot"][rephrasing_id]
    demo_block = build_demo_block(demos)
    return f"{instruction}\n{build_label_line(task)}\n\n{demo_block}\n\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_one_shot(task, rephrasing_id, demos):
    return prompt_n_shot(task, rephrasing_id, demos)


def prompt_few_shot_2(task, rephrasing_id, demos):
    return prompt_n_shot(task, rephrasing_id, demos)


def prompt_few_shot_3(task, rephrasing_id, demos):
    return prompt_n_shot(task, rephrasing_id, demos)


def prompt_role_based(task, rephrasing_id, demos=None):
    #role-based prompting assigns the model a lawyer role matched to the legal domain of the task.
    domain = CATEGORY_DOMAIN.get(task.task_type, "law")
    instruction = INSTRUCTION_REPHRASINGS["role_based"][rephrasing_id].format(domain=domain)
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_structured(task, rephrasing_id, demos=None):
    #structured prompting separates the task information into explicitly labelled fields before asking for the classification.
    instruction = INSTRUCTION_REPHRASINGS["structured"][rephrasing_id]
    return (f"Jurisdiction: {task.jurisdiction}\nPractice Area: {CATEGORY_DOMAIN.get(task.task_type, 'law')}\n"
            f"Relevant Facts: {task.context}\nConstraints: Choose exactly one of: {', '.join(task.label_options)}\n\n"
            f"{instruction}\nQuestion: {task.question}\n{build_final_answer_instruction()}")


def prompt_cot(task, rephrasing_id, demos=None):
    #chain-of-thought prompting explicitly asks the model to reason step by step before giving its final classification.
    instruction = INSTRUCTION_REPHRASINGS["cot"][rephrasing_id]
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


def prompt_legal_framework(task, rephrasing_id, demos, framework_name):
    #builds the prompts for IRAC, CRAC, CREAC, CLEO, TREACC and IREAC using the framework step sequences defined in config.py.
    steps = config.FRAMEWORK_STEPS[framework_name]
    steps_text = "; ".join(f"({i+1}) {s}" for i, s in enumerate(steps))
    instruction_variants = [
        f"Work through the following steps before answering: {steps_text}.",
        f"Analyze this using the {framework_name.upper()} method, addressing each part in order: {steps_text}.",
        f"Structure your reasoning according to these steps: {steps_text}.",
    ]
    #the legal frameworks also receive 3 equivalent instruction rephrasings, while keeping the actual framework steps unchanged.
    instruction = instruction_variants[rephrasing_id]
    return f"{instruction}\n{build_label_line(task)}\n{build_context_question_block(task)}\n{build_final_answer_instruction()}"


#maps each of the 13 strategy names to the function that constructs its prompt.
#the framework strategies all use the same prompt-building function but pass their own framework name to select the correct sequence of steps.
STRATEGY_FUNCTIONS = {
    "zero_shot": prompt_zero_shot,
    "one_shot": prompt_one_shot,
    "few_shot_2": prompt_few_shot_2,
    "few_shot_3": prompt_few_shot_3,
    "role_based": prompt_role_based,
    "structured": prompt_structured,
    "cot": prompt_cot,
    "irac": lambda t, r, d: prompt_legal_framework(t, r, d, "irac"),
    "crac": lambda t, r, d: prompt_legal_framework(t, r, d, "crac"),
    "creac": lambda t, r, d: prompt_legal_framework(t, r, d, "creac"),
    "cleo": lambda t, r, d: prompt_legal_framework(t, r, d, "cleo"),
    "treacc": lambda t, r, d: prompt_legal_framework(t, r, d, "treacc"),
    "ireac": lambda t, r, d: prompt_legal_framework(t, r, d, "ireac"),
}

#quick check that every strategy listed in config.py has a prompt function here, and that there are no extra strategies here that are missing from config.py.
assert set(STRATEGY_FUNCTIONS.keys()) == set(config.ALL_STRATEGIES), \
    "Mismatch between STRATEGY_FUNCTIONS and config.ALL_STRATEGIES!"


def build_prompt(task, strategy, rephrasing_id, task_id, demonstration_sets):
    """
    Builds the final prompt for the requested strategy.
    demonstration_sets stores the independently-selected demonstration sets for each task by k:
        {1: [...], 2: [...], 3: [...]}
    If the strategy needs demonstrations, the required k is looked up in config.py and that exact demonstration set is used.
    Strategies that do not need demonstrations are given None instead.
    """
    fn = STRATEGY_FUNCTIONS[strategy]

    if strategy in config.DEMO_REQUIRED_STRATEGIES:
        k = config.DEMO_REQUIRED_STRATEGIES[strategy]
        demos = demonstration_sets.get(task_id, {}).get(k, [])
        return fn(task, rephrasing_id, demos)

    return fn(task, rephrasing_id, None)