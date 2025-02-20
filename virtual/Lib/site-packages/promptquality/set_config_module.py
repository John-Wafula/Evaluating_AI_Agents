from os import getenv
from typing import Optional

from promptquality.constants.config import ConfigEnvironmentVariables
from promptquality.types.config import Config
from promptquality.utils.config import get_config_location


def set_config(console_url: Optional[str] = None) -> Config:
    """
    Set the config for `promptquality`.

    If the config file exists, and console_url is not passed, read it and return the
    config. Otherwise, set the default console URL and return the config.

    Parameters
    ----------
    console_url : Optional[str], optional
        URL to the Galileo console, by default None and we use the Galileo Cloud URL.

    Returns
    -------
    Config
        Config object for `promptquality`.
    """

    if not console_url and get_config_location().exists():
        config = Config.read()
    else:
        console_url = console_url or getenv(ConfigEnvironmentVariables.console_url)
        if not console_url:
            raise ValueError(
                "Please pass your Galileo console URL, or set the environment "
                "variable `GALILEO_CONSOLE_URL` to the Galileo Console URL."
            )
        config = Config(console_url=console_url)
    config.write()
    return config
