from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from promptquality.types.rows import PromptRow
from promptquality.utils.name import check_scorer_name

CustomMetricType = Union[float, int, bool, str, None]


class CustomScorer(BaseModel):
    name: str
    scorer_fn: Callable[[PromptRow], CustomMetricType] = Field(validation_alias="executor")
    aggregator_fn: Optional[Callable[[List[CustomMetricType], List[int]], Dict[str, CustomMetricType]]] = Field(
        default=None, validation_alias="aggregator"
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("name", mode="before")
    def validate_scorer_name(cls, name: str) -> str:
        return check_scorer_name(name)
