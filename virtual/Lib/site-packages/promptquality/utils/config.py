from pathlib import Path


def get_config_location() -> Path:
    return Path.home().joinpath(".galileo", "pq-config.json")
