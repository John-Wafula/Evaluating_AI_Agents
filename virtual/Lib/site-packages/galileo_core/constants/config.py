from enum import Enum


class ConfigEnvironmentVariables(str, Enum):
    console_url = "GALILEO_CONSOLE_URL"
    username = "GALILEO_USERNAME"
    password = "GALILEO_PASSWORD"
    api_key = "GALILEO_API_KEY"
    jwt_token = "GALILEO_JWT_TOKEN"
