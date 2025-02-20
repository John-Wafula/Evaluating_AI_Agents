from typing import Any, Dict, List, Optional

from pydantic import UUID4, BaseModel, ConfigDict, Field

from promptquality.types.pagination import (
    PaginationRequestMixin,
    PaginationResponseMixin,
)


class GetRowsRequest(PaginationRequestMixin):
    project_id: UUID4
    run_id: UUID4
    task_type: int

    def params(self) -> Dict[str, Any]:
        """
        Params to be passed to the API request.

        These are primarily the pagination parameters and task type.

        Returns
        -------
        Dict[str, Any]
            Params to be passed to the API request.
        """
        return self.model_dump(mode="json", exclude={"project_id", "run_id"})


class Metrics(BaseModel):
    uncertainty: Optional[float] = Field(default=None, validation_alias="hallucination")
    bleu: Optional[float] = None
    rouge: Optional[float] = None
    pii: Optional[List[str]] = Field(default=[])
    toxicity: Optional[float] = None
    correctness: Optional[float] = Field(default=None, validation_alias="factuality")
    correctness_explanation: Optional[str] = Field(default=None, validation_alias="factuality_explanation")
    context_adherence: Optional[float] = Field(default=None, validation_alias="groundedness")
    context_adherence_explanation: Optional[str] = Field(default=None, validation_alias="groundedness_explanation")
    latency: Optional[float] = None
    context_relevance: Optional[float] = None

    model_config = ConfigDict(extra="allow")


class PromptRow(BaseModel):
    index: int
    prompt: Optional[str] = None
    response: Optional[str] = None
    target: Optional[str] = None
    inputs: Dict[str, Optional[Any]] = Field(default_factory=dict)
    hallucination: Optional[float] = None
    bleu: Optional[float] = None
    rouge: Optional[float] = None
    cost: Optional[float] = None
    metrics: Metrics = Field(default_factory=Metrics)

    model_config = ConfigDict(extra="allow")


class PromptRows(PaginationResponseMixin):
    rows: List[PromptRow] = Field(default_factory=list)
