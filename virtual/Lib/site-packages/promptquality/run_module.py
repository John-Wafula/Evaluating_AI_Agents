from typing import List, Optional, Union

from pydantic import UUID4

from promptquality.constants.run import RunDefaults
from promptquality.constants.scorers import Scorers
from promptquality.get_metrics_module import get_run_metrics
from promptquality.helpers import (
    create_job,
    create_project,
    create_run,
    create_template,
    get_template_version_from_name,
    upload_custom_metrics,
    upload_dataset,
)
from promptquality.job_progress_module import job_progress
from promptquality.set_config_module import set_config
from promptquality.types.config import Config
from promptquality.types.custom_scorer import CustomScorer
from promptquality.types.customized_scorer import CustomizedChainPollScorer
from promptquality.types.registered_scorers import RegisteredScorer
from promptquality.types.run import (
    PromptMetrics,
    RunTag,
    ScorersConfiguration,
    TemplateVersion,
)
from promptquality.types.settings import Settings
from promptquality.utils.dataset import DatasetType
from promptquality.utils.name import ts_name
from promptquality.utils.scorer import bifurcate_scorers


def run(
    template: Union[str, TemplateVersion],
    dataset: Optional[Union[UUID4, DatasetType]] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    template_name: Optional[str] = None,
    scorers: Optional[List[Union[Scorers, CustomizedChainPollScorer, CustomScorer, RegisteredScorer, str]]] = None,
    generated_scorers: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
    run_tags: Optional[List[RunTag]] = None,
    wait: bool = True,
    silent: bool = False,
    scorers_config: ScorersConfiguration = ScorersConfiguration(),
    config: Optional[Config] = None,
) -> Optional[PromptMetrics]:
    """
    Create a prompt run.

    This function creates a prompt run that can be viewed on the Galileo console. The
    processing of the prompt run is asynchronous, so the function will return
    immediately. If the `wait` parameter is set to `True`, the function will block
    until the prompt run is complete.

    Additionally, all of the scorers are executed asynchronously in the background after
    the prompt run is complete, regardless of the value of the `wait` parameter.

    Parameters
    ----------
    template : Union[str, TemplateVersion]
        Template text or version information to use for the prompt run.
    dataset : Optional[DatasetType]
        Dataset to use for the prompt run.
    project_name : Optional[str], optional
        Project name to use, by default None which translates to a randomly generated
        name.
    run_name : Optional[str], optional
        Run name to use, by default None which translates to one derived from the
        project name, current timestamp and template version.
    template_name : Optional[str], optional
        Template name to use, by default None which translates to the project name.
    scorers : List[Union[Scorers, CustomScorer, RegisteredScorer, str]], optional
        List of scorers to use, by default None.
    settings : Optional[Settings], optional
        Settings to use, by default None which translates to the default settings.
    run_tags: Optional[List[RunTag]], optional,
        List of tags to attribute to a run, by default no tags will be added.
    wait : bool, optional
        Whether to wait for the prompt run to complete, by default True.
    silent : bool, optional
        Whether to suppress the console output, by default False.
    scorers_config : ScorersConfig, optional
        Can be used to enable or disable scorers. Can be used instead of scorers param,
        or can be used to disable default scorers.
    customized_scorers : Optional[List[CustomizedChainPollScorer]], optional
        List of customized GPT scorers to use, by default None.
    config : Optional[Config], optional
        Config to use, by default None which translates to the config being set
        automatically.

    Returns
    -------
    Optional[PromptMetrics]
        Metrics for the prompt run. These are only returned if the `wait` parameter is
        `True` for metrics that have been computed upto that point. Other metrics will
        be computed asynchronously.
    """
    config = config or set_config()
    # Create project.
    project = create_project(project_name, config)
    # Create template.
    if isinstance(template, str):
        template_response = create_template(
            template,
            project.id,
            # Use project name as template name if not provided.
            template_name=template_name or project.name,
            config=config,
        )
        template_name = template_response.name
        template_version_id = template_response.selected_version_id
    else:
        template_version = get_template_version_from_name(template, project.id, config)
        template_name = template.name
        template_version_id = template_version.id
    # Upload dataset.

    dataset_id = (
        upload_dataset(
            dataset,
            project.id,
            template_version_id,
            config,
        )
        if dataset is not None
        else None
    )
    # Run prompt.
    run_id = create_run(
        project.id,
        run_name=run_name or ts_name(prefix=f"{template_name}-v{template_version_id}"),
        run_tags=run_tags,
        config=config,
    )
    galileo_scorers, customized_scorers, custom_scorers, registered_scorers = bifurcate_scorers(scorers)
    scorers_config = scorers_config.merge_scorers(galileo_scorers)
    job_id = create_job(
        project.id,
        run_id,
        dataset_id,
        template_version_id,
        settings,
        scorers_config,
        registered_scorers,
        generated_scorers,
        customized_scorers,
        config,
    )
    if wait:
        job_progress(job_id, config)
    if not silent:
        print(f"🔭 View your prompt run on the Galileo console at: {config.current_run_url}")
    if custom_scorers:
        for scorer in custom_scorers:
            upload_custom_metrics(
                scorer,
                project_id=project.id,
                run_id=run_id,
                task_type=RunDefaults.prompt_evaluation_task_type,
                config=config,
            )
    return get_run_metrics(project_id=project.id, run_id=run_id, job_id=job_id, config=config)
