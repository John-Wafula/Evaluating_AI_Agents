from pydantic import UUID4

from galileo_core.constants.request_method import RequestMethod
from galileo_core.constants.routes import Routes
from galileo_core.helpers.logger import logger
from galileo_core.schemas.base_config import BaseConfig
from galileo_core.schemas.core.collaboration_role import CollaboratorRole
from galileo_core.schemas.core.user_project import UserProjectCollaboratorRequest, UserProjectCollaboratorResponse


def share_project_with_user(
    project_id: UUID4, user_id: UUID4, role: CollaboratorRole = CollaboratorRole.viewer
) -> UserProjectCollaboratorResponse:
    config = BaseConfig.get()
    logger.debug(f"Sharing project {project_id} with user {user_id} with role {role}...")
    response_dict = config.api_client.request(
        RequestMethod.POST,
        Routes.project_users.format(project_id=project_id),
        json=[UserProjectCollaboratorRequest(user_id=user_id, role=role).model_dump(mode="json")],
    )
    user_shared = [UserProjectCollaboratorResponse.model_validate(user) for user in response_dict]
    logger.debug(f"Shared project {project_id} with user {user_id} with role {role}.")
    return user_shared[0]
