from typing import List, Optional, Union

from galileo_core.schemas.shared.scorers.base_configs import GeneratedScorerConfig
from promptquality.constants.run import RunDefaults
from promptquality.constants.scorers import Scorers
from promptquality.helpers import (
    create_project,
    create_run,
    ingest_chain_rows,
    upload_custom_metrics,
)
from promptquality.job_progress_module import job_progress
from promptquality.set_config_module import set_config
from promptquality.types.chains.row import NodeRow
from promptquality.types.config import Config
from promptquality.types.custom_scorer import CustomScorer
from promptquality.types.customized_scorer import CustomizedChainPollScorer
from promptquality.types.registered_scorers import RegisteredScorer
from promptquality.types.run import RunTag, ScorersConfiguration
from promptquality.utils.scorer import bifurcate_scorers


def chain_run(
    rows: List[NodeRow],
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    scorers: Optional[List[Union[Scorers, CustomScorer, CustomizedChainPollScorer, RegisteredScorer, str]]] = None,
    generated_scorers: Optional[List[str]] = None,
    run_tags: Optional[List[RunTag]] = None,
    wait: bool = True,
    silent: bool = False,
    scorers_config: ScorersConfiguration = ScorersConfiguration(),
    config: Optional[Config] = None,
) -> None:
    if len(rows) < 1:
        raise ValueError("Chain run must have at least 1 row.")
    config = config or set_config()
    project = create_project(project_name, config)
    run_id = create_run(
        project.id, run_name=run_name, task_type=RunDefaults.prompt_chain_task_type, run_tags=run_tags, config=config
    )
    galileo_scorers, customized_scorers, custom_scorers, registered_scorers = bifurcate_scorers(scorers)
    scorers_config = scorers_config.merge_scorers(galileo_scorers)
    # If node_id is chain_id for another node then has_children set to True, otherwise set to False.
    ids_with_children = {row.chain_id for row in rows}
    for row in rows:
        row.has_children = row.node_id in ids_with_children

    ingestion_response = ingest_chain_rows(
        rows,
        project.id,
        run_id,
        scorers_config=scorers_config,
        registered_scorers=registered_scorers,
        customized_scorers=customized_scorers,
        generated_scorers=[GeneratedScorerConfig(name=generated_scorer) for generated_scorer in generated_scorers]
        if generated_scorers
        else None,
        config=config,
    )
    if wait:
        job_progress(ingestion_response.job_id, config=config)
    if not silent:
        print(f"🔭 View your prompt run on the Galileo console at: {config.current_run_url}")
    if custom_scorers:
        for scorer in custom_scorers:
            upload_custom_metrics(
                scorer,
                project_id=project.id,
                run_id=run_id,
                task_type=RunDefaults.prompt_chain_task_type,
                config=config,
            )
