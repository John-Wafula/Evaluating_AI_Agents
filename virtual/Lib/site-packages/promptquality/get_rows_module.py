from typing import List, Optional

from pydantic import UUID4

from promptquality.set_config_module import set_config
from promptquality.types.config import Config
from promptquality.types.pagination import PaginationDefaults
from promptquality.types.rows import GetRowsRequest, PromptRow, PromptRows


def get_rows(
    project_id: Optional[UUID4] = None,
    run_id: Optional[UUID4] = None,
    task_type: Optional[int] = None,
    config: Optional[Config] = None,
    starting_token: int = PaginationDefaults.starting_token,
    limit: int = PaginationDefaults.limit,
) -> List[PromptRow]:
    config = config or set_config()
    project_id = project_id or config.current_project_id
    if not project_id:
        raise ValueError("project_id must be provided")
    run_id = run_id or config.current_run_id
    if not run_id:
        raise ValueError("run_id must be provided")
    task_type = task_type or config.current_run_task_type
    if not task_type:
        raise ValueError("task_type must be provided")
    rows_json = config.api_client.get_rows(
        GetRowsRequest(
            project_id=project_id,
            run_id=run_id,
            task_type=task_type,
            starting_token=starting_token,
            limit=limit,
        )
    )
    rows_response = PromptRows.model_validate(rows_json)
    # If we paginated, return the rows + the rows from the next page.
    if rows_response.paginated:
        assert (
            rows_response.next_starting_token is not None
        ), "next_starting_token is None even though paginated is True."
        return rows_response.rows + get_rows(
            project_id=project_id,
            run_id=run_id,
            starting_token=rows_response.next_starting_token,
            limit=limit,
            task_type=task_type,
        )
    # If we didn't paginate, return the rows.
    else:
        return rows_response.rows
