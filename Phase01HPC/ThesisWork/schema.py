#this file defines the shared data structures used to represent LegalBench tasks, few-shot demonstrations, and the generation records saved during the experiment.

from dataclasses import dataclass, field
#dataclass is used here because these objects mainly store structured data. field is used to give the demonstration list a safe default value.

from typing import List, Optional
#List is used for fields that contain multiple values, while Optional is used for fields that may initially be None.

import Phase01HPC.ThesisWork.config as config
#imports the shared experiment configuration, particularly the required "Final Answer:" prefix used when demonstrations are rendered.


@dataclass
class LegalTask:
    """Represents one LegalBench task instance in the common format used by the rest of the pipeline."""
    task_id: str
    task_type: str          #the LegalBench reasoning category; one of the categories defined in config.py
    context: str
    question: str
    label_options: List[str] #the possible answer labels supplied for this task
    expected_output: str #the correct label for this instance
    jurisdiction: str = "US General"
    source_dataset: str = "LegalBench"


@dataclass
class Demonstration:
    """Represents one fixed example used in the demonstration-based prompting strategies."""
    context: str
    question: str
    label: str

    def render(self) -> str:
        #turns the demonstration into the text format that will actually be inserted into the prompt.
        #if the task has a question, both the context and question are included; otherwise only the context is included.
        if self.question:
            return f"Context: {self.context}\nQuestion: {self.question}\n{config.FINAL_ANSWER_PREFIX} {self.label}"
        return f"Context: {self.context}\n{config.FINAL_ANSWER_PREFIX} {self.label}"


@dataclass
class TaskDemonstrationSet:
    """Stores the fixed demonstration set selected for one task so that the same examples can be reused throughout the experiment."""
    task_id: str
    demonstrations: List[Demonstration] = field(default_factory=list)


@dataclass
class GenerationRecord:
    """Stores everything recorded for one model generation before it is written to the raw generation outputs."""
    task_id: str
    category: str
    strategy: str
    rephrasing_id: int      #which of the 3 instruction rephrasings was used: 0, 1 or 2
    run_id: int              #which repeated run produced this generation: 0 through N_RUNS-1
    model_name: str
    prompt_text: str
    raw_output: str
    parsed_answer: Optional[str] = None #the answer extracted from the raw model output; remains None if parsing fails
    is_correct: Optional[bool] = None #whether the parsed answer matches the expected label
    timestamp: Optional[str] = None   #the ISO 8601 timestamp recorded when the generation is produced