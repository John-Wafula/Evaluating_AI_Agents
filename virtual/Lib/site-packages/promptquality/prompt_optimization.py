from pathlib import Path
from typing import Optional

from pydantic import UUID4

from promptquality.constants.run import RunDefaults
from promptquality.helpers import (
    create_project,
    create_prompt_optimization_job,
    create_run,
    create_template,
    get_job_status,
    upload_dataset,
)
from promptquality.set_config_module import set_config
from promptquality.types.config import Config
from promptquality.types.prompt_optimization import (
    PromptOptimizationConfiguration,
    PromptOptimizationResults,
)
from promptquality.types.run import GetMetricsRequest
from promptquality.utils.name import ts_name


def optimize_prompt(
    prompt_optimization_config: PromptOptimizationConfiguration,
    dataset: Path,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Config] = None,
) -> UUID4:
    """
    Optimize a prompt for a given task.

    This function takes a prompt and a list of evaluation criteria, and optimizes the
    prompt for the given task. The function uses the OpenAI API to generate and evaluate
    prompts, and returns the best prompt based on the evaluation criteria.

    Parameters
    ----------
    prompt_optimization_config : PromptOptimizationConfiguration
        Configuration for the prompt optimization job.
    dataset : Path
        Path to the training dataset.
    project_name : Optional[str], optional
        Name of the project, by default None. If None we will generate a name.
    run_name : Optional[str], optional
        Name of the run, by default None. If None we will generate a name.
    config : Optional[Config], optional
        pq config object, by default None. If None we will use the default config.

    Returns
    -------
    job_id: UUID4
        Unique identifier required to fetch Prompt Optimization results.
    """
    config = config or set_config()

    project = create_project(project_name, config)
    template_response = create_template(
        prompt_optimization_config.prompt,
        project.id,
        # Use project name as template name if not provided.
        template_name=project.name,
        config=config,
    )
    dataset_id = upload_dataset(
        dataset,
        project.id,
        template_response.selected_version_id,
        config,
    )
    run_id = create_run(
        project.id,
        run_name=run_name or ts_name(prefix=f"{template_response.name}-v{template_response.selected_version.version}"),
        task_type=RunDefaults.prompt_optimization_task_type,
        config=config,
    )

    job_id = create_prompt_optimization_job(
        prompt_optimization_configuration=prompt_optimization_config,
        project_id=project.id,
        run_id=run_id,
        train_dataset_id=dataset_id,
        config=config,
    )

    return job_id


def fetch_prompt_optimization_result(
    job_id: Optional[UUID4] = None,
    config: Optional[Config] = None,
) -> PromptOptimizationResults:
    """
    Fetch the prompt optimization results.

    Parameters
    ----------
    job_id : UUID4
        Unique identifier required to fetch Prompt Optimization results.

    Returns
    -------
    PromptOptimizationResults
        - best_prompt: The best prompt based on the evaluation criteria.
        - train_results: List of epoch results for the training dataset.
            Sorted by epoch ascending.
        - val_results: List of epoch results for the validation dataset.
            Sorted by epoch ascending.
    """
    config = config or set_config()
    job_id = job_id or config.current_prompt_optimization_job_id
    if job_id is None:
        raise ValueError("job_id is required.")

    job = get_job_status(job_id, config)
    project_id, run_id = job.project_id, job.run_id

    metrics_request = GetMetricsRequest(project_id=project_id, run_id=run_id)
    all_metrics = config.api_client.get_metrics(metrics_request)
    if not all_metrics:
        epoch = 0
    else:
        epoch = max([metric["epoch"] for metric in all_metrics])
    for metric in all_metrics:
        if metric["key"] == "prompt_optimization_optimized":
            return PromptOptimizationResults(
                best_prompt=metric["extra"]["best_prompt"],
                finished_computing=True,
                epoch=-1,
            )

    return PromptOptimizationResults(
        best_prompt="",
        epoch=epoch,
        finished_computing=False,
    )
