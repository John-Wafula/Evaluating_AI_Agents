from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import Field
from typing_extensions import Annotated

from galileo_core.schemas.shared.workflows.node_type import NodeType
from galileo_core.schemas.shared.workflows.step import (
    BaseStep,
    BaseStepWithChildren,
    LlmStep,
    LlmStepAllowedIOType,
    RetrieverStep,
    RetrieverStepAllowedOutputType,
    StepIOType,
    ToolStep,
)


class SpanWithParentStep(BaseStep):
    parent: Optional["StepWithChildSpans"] = Field(
        default=None, description="Parent node of the current node. For internal use only.", exclude=True
    )


class StepWithChildSpans(BaseStepWithChildren):
    spans: List["Span"] = Field(default_factory=list, description="Child spans.")

    def children(self) -> Sequence[BaseStep]:
        return self.spans

    def add_child(self, *spans: "Span") -> None:
        for span in spans:
            span.parent = self
            self.spans.append(span)

    def add_llm_span(
        self,
        input: LlmStepAllowedIOType,
        output: LlmStepAllowedIOType,
        model: str,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        status_code: Optional[int] = None,
        ground_truth: Optional[str] = None,
    ) -> "LlmSpan":
        """
        Add a new llm span to the current workflow.

        Parameters:
        ----------
            input: LlmSpanAllowedIOType: Input to the node.
            output: LlmSpanAllowedIOType: Output of the node.
            model: str: Model used for this span.
            tools: Optional[Sequence[Dict[str, Any]]]: List of available tools passed to LLM on invocation.
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
            input_tokens: Optional[int]: Number of input tokens.
            output_tokens: Optional[int]: Number of output tokens.
            total_tokens: Optional[int]: Total number of tokens.
            temperature: Optional[float]: Temperature used for generation.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            LlmSpan: The created span.
        """
        span = LlmSpan(
            parent=self,
            input=input,
            output=output,
            model=model,
            tools=tools,
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            temperature=temperature,
            status_code=status_code,
            ground_truth=ground_truth,
        )
        self.add_child(span)
        return span

    def add_retriever_span(
        self,
        input: StepIOType,
        documents: RetrieverStepAllowedOutputType,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        status_code: Optional[int] = None,
    ) -> "RetrieverSpan":
        """
        Add a new retriever span to the current workflow.

        Parameters:
        ----------
            input: SpanIOType: Input to the node.
            documents: Union[List[str], List[Dict[str, str]], List[Document]]: Documents retrieved from the retriever.
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            RetrieverSpan: The created span.
        """
        span = RetrieverSpan(
            parent=self,
            input=input,
            output=documents,
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
            status_code=status_code,
        )
        self.add_child(span)
        return span

    def add_tool_span(
        self,
        input: StepIOType,
        output: StepIOType,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        status_code: Optional[int] = None,
    ) -> "ToolSpan":
        """
        Add a new tool span to the current workflow.

        Parameters:
        ----------
            input: SpanIOType: Input to the node.
            output: SpanIOType: Output of the node.
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            ToolSpan: The created span.
        """
        span = ToolSpan(
            parent=self,
            input=input,
            output=output,
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
            status_code=status_code,
        )
        self.add_child(span)
        return span

    def add_workflow_span(
        self,
        input: StepIOType,
        output: Optional[StepIOType] = None,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "WorkflowSpan":
        """
        Add a nested workflow span to the workflow. This is useful when you want to create a nested workflow within the
        current workflow. The next span you add will be a child of this workflow. To span out of the nested workflow,
        use conclude_workflow().

        Parameters:
        ----------
            input: SpanIOType: Input to the node.
            output: Optional[SpanIOType]: Output of the node. This can also be set on conclude_workflow().
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
        Returns:
        -------
            WorkflowSpan: The created span.
        """
        span = WorkflowSpan(
            parent=self,
            input=input,
            output=output or "",
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
        )
        self.add_child(span)
        return span

    def conclude(
        self, output: Optional[StepIOType] = None, duration_ns: Optional[int] = None, status_code: Optional[int] = None
    ) -> Optional["StepWithChildSpans"]:
        """
        Conclude the workflow by setting the output of the current node. In the case of nested workflows, this will
        point the workflow back to the parent of the current workflow.

        Parameters:
        ----------
            output: Optional[SpanIOType]: Output of the node.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            Optional[SpanWithChildren]: The parent of the current workflow. None if no parent exists.
        """
        self.output = output or self.output
        self.status_code = status_code
        if duration_ns is not None:
            self.duration_ns = duration_ns

        if isinstance(self, SpanWithParentStep):
            return self.parent
        return None


class Trace(StepWithChildSpans):
    type: Literal[NodeType.trace] = Field(
        default=NodeType.trace, description="Type of the span. By default, it is set to trace."
    )


class WorkflowSpan(StepWithChildSpans, SpanWithParentStep):
    type: Literal[NodeType.workflow] = Field(
        default=NodeType.workflow, description="Type of the span. By default, it is set to workflow."
    )


class LlmSpan(SpanWithParentStep, LlmStep):
    type: Literal[NodeType.llm] = Field(
        default=NodeType.llm, description="Type of the step. By default, it is set to llm."
    )


class RetrieverSpan(SpanWithParentStep, RetrieverStep):
    type: Literal[NodeType.retriever] = Field(
        default=NodeType.retriever, description="Type of the step. By default, it is set to retriever."
    )


class ToolSpan(SpanWithParentStep, ToolStep):
    type: Literal[NodeType.tool] = Field(
        default=NodeType.tool, description="Type of the step. By default, it is set to tool."
    )


Span = Annotated[Union[WorkflowSpan, LlmSpan, RetrieverSpan, ToolSpan], Field(discriminator="type")]
