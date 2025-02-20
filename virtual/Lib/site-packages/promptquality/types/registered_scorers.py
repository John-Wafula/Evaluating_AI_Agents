from typing import List, Optional

from pydantic import UUID4, BaseModel, ConfigDict, Field

from promptquality.types.pagination import PaginationResponseMixin


class RegisteredScorer(BaseModel):
    # The aliases are what the API returns as a part of the response.
    id: UUID4 = Field(alias="registered_scorer_id")
    name: str = Field(alias="metric_name")
    score_type: Optional[str] = None
    scoreable_node_types: Optional[List[str]] = None

    model_config = ConfigDict(populate_by_name=True)


class ListRegisteredScorers(PaginationResponseMixin):
    scorers: List[RegisteredScorer]
