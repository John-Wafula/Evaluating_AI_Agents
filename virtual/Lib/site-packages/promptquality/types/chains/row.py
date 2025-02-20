from json import dumps
from time import time_ns
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from warnings import warn

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from galileo_core.schemas.shared.scorers.base_configs import GeneratedScorerConfig
from galileo_core.schemas.shared.workflows.node_type import NodeType
from promptquality.constants.row import GALILEO_PROTECT_NODE_NAME
from promptquality.types.customized_scorer import CustomizedChainPollScorer
from promptquality.types.registered_scorers import RegisteredScorer
from promptquality.types.run import JobInfoMixin, ScorersConfiguration


class NodeRow(BaseModel):
    """
    Chains are constructed of `NodeRow`s. Each NodeRow represents a node in the chain and are modeled as a tree.

    Each chain has a root node, which is the first node in the chain. Each non-root node in the chain has a parent node.
    Parent nodes are necessarily chain nodes.

    The required fields for a chain row are `node_id`, `node_type`, `chain_root_id`, and `step`. The remaining fields
    are optional and are populated as the chain is executed.
    """

    node_id: UUID = Field(description="ID of that node in the chain. This maps to `run_id` from `langchain`.")
    node_type: NodeType = Field(description="Type of node in the chain.")
    node_name: Optional[str] = Field(default=None, description="Name of the node in the chain.")
    node_input: str = Field(default="", description="Stringified input to the node in the chain.")
    node_output: str = Field(default="", description="Stringified output from the node in the chain.")
    tools: Optional[str] = Field(
        default=None, description="Stringified list of tools available to the node in the chain."
    )
    # Chain fields.
    chain_root_id: UUID = Field(description="ID of the root node in the chain.")
    step: int = Field(
        description="Step in the chain. This is always increasing. The root node is step 1, with other nodes incrementing from there."
    )
    chain_id: Optional[UUID] = Field(
        default=None,
        description="ID of the parent node of the current node. This maps to `parent_run_id` from `langchain`.",
    )
    has_children: bool = Field(default=False, description="Indicates whether a node has 1 or more child nodes")
    # Inputs and prompt.
    inputs: Dict = Field(default_factory=dict, description="Inputs to the node, as key-value pairs.")
    prompt: Optional[str] = Field(default=None, description="Prompt for the node.")
    # Response fields.
    response: Optional[str] = Field(default=None, description="Response received after the node's execution.")
    creation_timestamp: int = Field(default_factory=time_ns, description="Timestamp when the node was created.")
    finish_reason: str = Field(default="", description="Reason for the node's completion.")
    latency: Optional[int] = Field(default=None, description="Latency of the node's execution in nanoseconds.")
    query_input_tokens: int = Field(default=0, description="Number of tokens in the query input.")
    query_output_tokens: int = Field(default=0, description="Number of tokens in the query output.")
    query_total_tokens: int = Field(default=0, description="Total number of tokens in the query.")
    # Metadata.
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters passed to the node.")
    target: Optional[str] = Field(
        default=None,
        description="Target output for a workflow. This is used for calculating BLEU and ROUGE scores, and only applicable at the root node level.",
    )

    # Ignore extra fields.
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    @field_validator("step", mode="before")
    def validate_step_on_root(cls, value: int, info: ValidationInfo) -> int:
        if info.data.get("node_id") == info.data.get("chain_root_id") and value != 0:
            raise ValueError("Root nodes should always be step 0.")
        return value

    @field_validator("chain_id", mode="before")
    def validate_chain_id(cls, value: Optional[UUID], info: ValidationInfo) -> Optional[UUID]:
        if value == info.data.get("chain_root_id") and info.data.get("step") == 0:
            raise ValueError("Chain ID cannot be the same as the chain root ID for the root node.")
        elif value == info.data.get("node_id"):
            raise ValueError("Chain ID cannot match node ID.")
        return value

    @field_validator("target", mode="before")
    def warn_target(cls, value: Optional[str], info: ValidationInfo) -> Optional[str]:
        if (value is not None) and (info.data.get("chain_root_id") != info.data.get("node_id")):
            warn("Target is set for a non-root node. BLEU and ROUGE scores will not be calculated on this node.")
        return value

    @classmethod
    def for_retriever(
        cls,
        query: str,
        documents: List[str],
        root_id: UUID,
        step: int = 1,
        id: Optional[UUID] = None,
        name: Optional[str] = None,
        latency: Optional[int] = None,
    ) -> "NodeRow":
        return cls(
            node_id=id or uuid4(),
            chain_root_id=root_id,
            chain_id=root_id,
            step=step,
            node_type=NodeType.retriever,
            node_name=name,
            node_input=query,
            node_output=dumps([dict(page_content=document) for document in documents]),
            latency=latency,
        )

    @classmethod
    def for_llm(
        cls,
        prompt: str,
        response: str,
        root_id: Optional[UUID] = None,
        step: int = 1,
        id: Optional[UUID] = None,
        name: Optional[str] = None,
        target: Optional[str] = None,
        latency: Optional[int] = None,
    ) -> "NodeRow":
        root_id = root_id or uuid4()
        return cls(
            node_id=root_id if step == 0 else id or uuid4(),
            chain_root_id=root_id,
            chain_id=root_id if step > 0 else None,
            step=step,
            node_type=NodeType.llm,
            node_name=name,
            prompt=prompt,
            node_input=prompt,
            response=response,
            node_output=response,
            target=target,
            latency=latency,
        )

    @classmethod
    def for_protect(
        cls,
        payload: str,
        response: str,
        root_id: Optional[UUID] = None,
        step: int = 1,
        id: Optional[UUID] = None,
        latency: Optional[int] = None,
    ) -> "NodeRow":
        root_id = root_id or uuid4()
        return cls(
            node_id=id or uuid4(),
            chain_root_id=root_id,
            chain_id=root_id,
            step=step,
            node_type=NodeType.tool,
            node_name=GALILEO_PROTECT_NODE_NAME,
            node_input=payload,
            node_output=response,
            latency=latency,
        )


class ChainIngestRequest(BaseModel):
    rows: List[NodeRow] = Field(default_factory=list)
    prompt_scorers_configuration: Optional[ScorersConfiguration] = Field(default=None, validate_default=True)
    prompt_registered_scorers_configuration: Optional[List[RegisteredScorer]] = Field(
        default=None, validate_default=True
    )
    prompt_customized_scorers_configuration: Optional[List[CustomizedChainPollScorer]] = Field(
        default=None, validate_default=True
    )
    generated_scorers: Optional[List[GeneratedScorerConfig]] = Field(default=None, validate_default=True)

    @field_validator("rows", mode="before")
    def sort_rows(cls, rows: List[NodeRow]) -> List[NodeRow]:
        """Sort rows by step number while maintaining the chain root ID order."""
        root_ids: List[UUID] = []
        for row in rows:
            if row.chain_root_id not in root_ids:
                root_ids.append(row.chain_root_id)
        return sorted(rows, key=lambda row: (root_ids.index(row.chain_root_id), row.step))


class ChainIngestResponse(JobInfoMixin):
    num_rows: int
    message: str
