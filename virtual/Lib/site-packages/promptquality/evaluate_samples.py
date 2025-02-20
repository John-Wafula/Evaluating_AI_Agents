from typing import Optional

from pydantic import UUID4

from promptquality.helpers import get_project_from_name, get_run_from_name
from promptquality.set_config_module import set_config
from promptquality.types.evaluate_samples import EvaluateSamples


def get_evaluate_samples(
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    project_id: Optional[UUID4] = None,
    run_id: Optional[UUID4] = None,
) -> EvaluateSamples:
    """
    Get the evaluate samples for a run in a project. Must pass either project_name or project_id and either run_name or
    run_id. If both are passed we default to the id.

    Parameters:
    ----------
        project_name: Optional[str]: The name of the project.
        run_name: Optional[str]: The name of the run.
        project_id: Optional[UUID4]: The id of the project.
        run_id: Optional[UUID4]: The id of the run.
    Returns:
    -------
        EvaluateSamples: The evaluate samples for the run.
        For workflows each sub node is nested within the base sample.
    """
    if project_id is None:
        if not project_name:
            raise ValueError("project_name or project_id must be provided")
        project = get_project_from_name(project_name)
        if project is None:
            raise ValueError(f"Project {project_name} not found")
        project_id = project.id
    if run_id is None:
        if not run_name:
            raise ValueError("run_name or run_id must be provided")
        run = get_run_from_name(run_name, project_id)
        if run is None:
            raise ValueError(f"Run {run_name} not found")
        run_id = run.id
    config = set_config()
    samples_list = config.api_client.get_evaluate_samples(
        project_id=project_id,
        run_id=run_id,
    )
    return EvaluateSamples(samples=samples_list)
