from typing import Optional

from pydantic import UUID4

from promptquality.helpers import get_project, get_project_from_name, get_templates
from promptquality.types.run import BaseTemplateResponse


def get_template(
    project_name: Optional[str] = None, project_id: Optional[UUID4] = None, template_name: Optional[str] = None
) -> BaseTemplateResponse:
    """
    Get a template for a specific project.

    Parameters
    ----------
    project_name : Optional[str]
        Project name.
    project_id : Optional[UUID4]
        Project ID.
    template_name : Optional[str]
        Template name.

    Returns
    -------
    BaseTemplateResponse
        Template response.
    """
    if project_name:
        project = get_project_from_name(project_name=project_name)
    elif project_id:
        project = get_project(project_id)
    else:
        raise Exception("Either project_id or project_name must be provided.")

    if not project:
        raise ValueError("Project not found.")

    if not template_name:
        raise Exception("template_name must be provided.")

    templates = get_templates(project.id)
    requested_template = None
    for template in templates:
        if template.name == template_name:
            requested_template = template
    if not requested_template:
        raise Exception(f"Template {template_name} not found.")
    return requested_template
