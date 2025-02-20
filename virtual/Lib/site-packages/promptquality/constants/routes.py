from types import SimpleNamespace

# We cannot use Enum here because `mypy` doesn't like it when we format those routes
# into strings. https://github.com/python/mypy/issues/15269
Routes = SimpleNamespace(
    healthcheck="healthcheck",
    login="login",
    api_key_login="login/api_key",
    get_token="get-token",
    current_user="current_user",
    projects="projects",
    project="projects/{project_id}",
    all_projects="projects/all",
    templates="projects/{project_id}/templates",
    template="projects/{project_id}/templates/{template_id}",
    versions="projects/{project_id}/templates/{template_id}/versions",
    version="projects/{project_id}/templates/{template_id}/versions/{version}",
    version_from_name="projects/{project_id}/templates/versions",
    dataset="projects/{project_id}/prompt_datasets",
    runs="projects/{project_id}/runs",
    rows="projects/{project_id}/runs/{run_id}/prompts/rows",
    export_rows="projects/{project_id}/runs/{run_id}/prompts/export_prompt_dataset",
    jobs="jobs",
    prompt_run="projects/{project_id}/prompts/runs/{run_id}",
    run_jobs="projects/{project_id}/runs/{run_id}/jobs",
    metrics="projects/{project_id}/runs/{run_id}/metrics",
    user_submitted_metrics=("projects/{project_id}/runs/{run_id}/prompts/scorers/user_submitted"),
    integrations="integrations/{integration_name}",
    registered_scorers="registered-scorers",
    registered_scorer="registered-scorers/{registered_scorer_id}",
    chain_rows="projects/{project_id}/runs/{run_id}/chains/ingest",
)
