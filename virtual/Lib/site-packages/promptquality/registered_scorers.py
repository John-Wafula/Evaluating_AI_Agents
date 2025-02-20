from pathlib import Path
from typing import List, Optional, Union

from pydantic import UUID4

from promptquality.set_config_module import set_config
from promptquality.types.config import Config
from promptquality.types.registered_scorers import (
    ListRegisteredScorers,
    RegisteredScorer,
)
from promptquality.utils.logger import logger
from promptquality.utils.name import check_scorer_name


def register_scorer(
    scorer_name: str, scorer_file: Union[str, Path], config: Optional[Config] = None
) -> RegisteredScorer:
    config = config or set_config()
    scorer_file = Path(scorer_file)
    if not scorer_file.exists():
        raise FileNotFoundError(f"Scorer file {scorer_file} does not exist.")
    check_scorer_name(scorer_name)
    response_dict = config.api_client.register_scorer(scorer_name, scorer_file)

    score_type = response_dict.get("score_type")
    score_type = score_type if score_type else "None"

    scoreable_node_types = response_dict.get("scoreable_node_types")
    scoreable_node_types = ", ".join(scoreable_node_types) if scoreable_node_types else "None"

    print(
        f"🔢 Registered scorer '{scorer_name}'. Score type: {score_type}, scorable node types: {scoreable_node_types}."
    )
    scorer = RegisteredScorer.model_validate(response_dict)
    logger.debug(f"Registered scorer {scorer.name} with id {scorer.id}.")
    return scorer


def list_registered_scorers(config: Optional[Config] = None) -> List[RegisteredScorer]:
    config = config or set_config()
    response_dict = config.api_client.list_registered_scorers()
    scorers = ListRegisteredScorers.model_validate(response_dict)
    logger.debug(f"Found {len(scorers.scorers)} registered scorers.")
    return scorers.scorers


def delete_registered_scorer(scorer_id: UUID4, config: Optional[Config] = None) -> None:
    config = config or set_config()
    config.api_client.delete_registered_scorer(scorer_id)
    logger.debug(f"Deleted scorer with id {scorer_id}.")
