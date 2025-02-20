from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomizedScorerName(str, Enum):
    chunk_attribution_utilization_plus = "_customized_chunk_attribution_utilization_gpt"
    completeness_plus = "_customized_completeness_gpt"
    context_adherence_plus = "_customized_groundedness"
    correctness = "_customized_factuality"
    instruction_adherence = "_customized_instruction_adherence"
    tool_errors = "_customized_tool_error_rate"
    tool_selection_quality = "_customized_tool_selection_quality"
    action_advancement = "_customized_agentic_workflow_success"


class CustomizedScorer(BaseModel):
    scorer_name: CustomizedScorerName = Field(..., description="Name of the customized scorer.")
    model_alias: Optional[str] = Field(default=None, description="Model alias to use for scoring.")
    num_judges: Optional[int] = Field(default=None, ge=1, le=10, description="Number of judges for the scorer.")

    model_config = ConfigDict(
        # Avoid Pydantic's protected namespace warning since we want to use
        # `model_alias` as a field name.
        protected_namespaces=(),
    )
