from functools import partial
from typing import List, Optional

from pydantic import UUID4

from galileo_core.constants.request_method import RequestMethod
from galileo_core.constants.routes import Routes
from galileo_core.helpers.logger import logger
from galileo_core.helpers.pagination import paginated_request
from galileo_core.schemas.base_config import BaseConfig
from galileo_core.schemas.core.user import InviteUsersRequest, UpdateUserRoleRequest, User
from galileo_core.schemas.core.user_role import UserRole


def get_current_user() -> User:
    """
    Get the current user.

    Returns
    -------
    User
        Current user.
    """
    config = BaseConfig.get()
    logger.debug("Getting current user...")
    response_dict = config.api_client.request(RequestMethod.GET, Routes.current_user)
    user = User.model_validate(response_dict)
    logger.debug(f"Got current user {user.email}.")
    return user


def invite_users(emails: List[str], role: UserRole = UserRole.user, group_ids: Optional[List[UUID4]] = None) -> None:
    """
    Invite users.

    Parameters
    ----------
    emails : List[str]
        List of emails to invite.
    role : UserRole, optional
        Roles to grant invited users, by default UserRole.user
    group_ids : Optional[List[UUID4]], optional
        Group IDs to add the users to, by default None, which means they are not added to any group.
    """
    config = BaseConfig.get()
    group_ids = group_ids or list()
    request = InviteUsersRequest(emails=emails, role=role, group_ids=group_ids)
    logger.debug(f"Inviting users {request.emails} with role {request.role}...")
    config.api_client.request(RequestMethod.POST, Routes.invite_users, json=request.model_dump(mode="json"))
    logger.debug(f"Invited users {request.emails} with role {request.role}.")


def list_users() -> List[User]:
    """
    List all users.

    Returns
    -------
    List[User]
        List of all users.
    """
    config = BaseConfig.get()
    logger.debug("Listing users...")
    all_users = paginated_request(partial(config.api_client.request, RequestMethod.GET, Routes.users), "users")
    users = [User.model_validate(user) for user in all_users]
    logger.debug(f"Listed all users, found {len(users)} users.")
    return users


def update_user(user_id: UUID4, role: UserRole) -> User:
    """
    Update user.

    Parameters
    ----------
    user_id : User.id
        User ID to update.
    role : UserRole
        New role to assign to the user.

    Returns
    -------
    User
        Updated user.
    """
    config = BaseConfig.get()
    logger.debug(f"Updating user {str(user_id)} to {role}...")

    request = UpdateUserRoleRequest(role=role)
    response_dict = config.api_client.request(
        RequestMethod.PUT, Routes.update_user.format(user_id=user_id), json=request.model_dump(mode="json")
    )
    user = User.model_validate(response_dict)
    logger.debug(f"Updated user {str(user_id)} role to {user.role.value}.")
    return user
