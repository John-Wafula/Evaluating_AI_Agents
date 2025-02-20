from datetime import datetime
from re import match

SCORER_NAME_REGEX = r"^[\w -]+$"


def ts_name(prefix: str) -> str:
    ts = datetime.now()
    ts_string = ts.strftime("%b_%d_%H_%M_%S")
    return f"{prefix}-{ts_string}"


def check_scorer_name(name: str) -> str:
    """
    Check if name contains only letters, numbers, space, - and _.
    """
    if not bool(match(SCORER_NAME_REGEX, name)):
        raise ValueError("Scorer name cannot contain special characters, only letters, numbers, space, - and _.")
    return name
