from typing import Optional

from promptquality.set_config_module import set_config
from promptquality.types.config import Config


def login(console_url: Optional[str] = None) -> Config:
    """
    Login to Galileo Environment.

    By default, this will login to Galileo Cloud but can be used to login to the enterprise version of Galileo by
    passing in the console URL for the environment.
    """
    config = set_config(console_url)
    config.login()
    return config
