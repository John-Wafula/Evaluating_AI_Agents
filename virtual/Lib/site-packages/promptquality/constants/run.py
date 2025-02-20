from enum import Enum
from types import SimpleNamespace

RunDefaults = SimpleNamespace(
    project_type="prompt_evaluation",
    prompt_run_job_name="prompt_run",
    prompt_scorer_job_name="prompt_scorer",
    prompt_evaluation_task_type=7,
    prompt_chain_task_type=12,
    prompt_optimization_task_type=14,
)


class TagType(str, Enum):
    GENERIC = "generic"
    RAG = "rag"
