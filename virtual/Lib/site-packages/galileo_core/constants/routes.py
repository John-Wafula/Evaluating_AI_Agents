from enum import Enum


class Routes(str, Enum):
    healthcheck = "healthcheck"
    username_login = "login"
    api_key_login = "login/api_key"
    get_token = "get-token"
    current_user = "current_user"
    projects = "projects"
    project = "projects/{project_id}"
    project_groups = "projects/{project_id}/groups"
    project_users = "projects/{project_id}/users"
    groups = "groups"
    group_members = "groups/{group_id}/members"
    invite_users = "invite_users"
    users = "users"
    create_api_key = "users/api_keys"
    get_api_keys = "users/{user_id}/api_keys"
    delete_api_key = "users/api_keys/{api_key_id}"
    update_user = "users/{user_id}"
    datasets = "datasets"
    dataset_content = "datasets/{dataset_id}/content"
