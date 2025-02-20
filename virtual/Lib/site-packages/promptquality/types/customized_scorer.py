from typing import Optional

from pydantic import field_validator

from galileo_core.schemas.shared.customized_scorer import CustomizedScorer
from promptquality.constants.models import Models


class CustomizedChainPollScorer(CustomizedScorer):
    model_alias: Optional[Models] = None

    @field_validator("model_alias", mode="before")
    def validate_model_alias(cls, value: Optional[Models]) -> Optional[Models]:
        if value is not None:
            if value not in Models.for_customized_scorers():
                raise ValueError(f"Model {value} is not a valid model for customized scorers.")
        return value
