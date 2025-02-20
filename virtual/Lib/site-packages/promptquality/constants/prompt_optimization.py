from types import SimpleNamespace

from promptquality.constants.run import RunDefaults

DEFAULT_TEMPERATURE = 1
DEFAULT_MAX_TOKENS = 4096
PromptOptimizationDefaults = SimpleNamespace(
    project_type=RunDefaults.project_type,
    prompt_optimization_job_name="prompt_optimization",
    prompt_evaluation_task_type=RunDefaults.prompt_evaluation_task_type,
)
