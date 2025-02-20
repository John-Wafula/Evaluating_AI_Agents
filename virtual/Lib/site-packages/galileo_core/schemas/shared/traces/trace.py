from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from galileo_core.schemas.shared.traces.types import (
    LlmSpan,
    RetrieverSpan,
    StepWithChildSpans,
    ToolSpan,
    Trace,
    WorkflowSpan,
)
from galileo_core.schemas.shared.workflows.step import LlmStepAllowedIOType, RetrieverStepAllowedOutputType, StepIOType


class Traces(BaseModel):
    traces: List[Trace] = Field(default_factory=list, description="List of traces.")
    current_parent: Optional[StepWithChildSpans] = Field(default=None, description="Current parent context.")

    def add_trace(
        self,
        input: StepIOType,
        output: Optional[StepIOType] = None,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        ground_truth: Optional[str] = None,
    ) -> Trace:
        """
        Create a new trace and add it to the list of traces.
        Simple usage:
        ```
        my_traces.add_trace("input")
        my_traces.add_llm_span("input", "output", model="<my_model>")
        my_traces.conclude("output")
        ```
        Parameters:
        ----------
            input: StepIOType: Input to the node.
            output: Optional[str]: Output of the node.
            name: Optional[str]: Name of the trace.
            duration_ns: Optional[int]: Duration of the trace in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the trace's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this trace.
            ground_truth: Optional[str]: Ground truth, expected output of the trace.
        Returns:
        -------
            Trace: The created trace.
        """
        trace = Trace(
            input=input,
            output=output or "",
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
            ground_truth=ground_truth,
        )
        self.traces.append(trace)
        self.current_parent = trace
        return trace

    def add_single_span_trace(
        self,
        input: LlmStepAllowedIOType,
        output: LlmStepAllowedIOType,
        model: str,
        tools: Optional[List[Dict]] = None,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        ground_truth: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> Trace:
        """
        Create a new trace with a single span and add it to the list of traces.

        Parameters:
        ----------
            input: LlmStepAllowedIOType: Input to the node.
            output: LlmStepAllowedIOType: Output of the node.
            model: str: Model used for this span. Feedback from April: Good docs about what model names we use.
            tools: Optional[List[Dict]]: List of available tools passed to LLM on invocation.
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
            input_tokens: Optional[int]: Number of input tokens.
            output_tokens: Optional[int]: Number of output tokens.
            total_tokens: Optional[int]: Total number of tokens.
            temperature: Optional[float]: Temperature used for generation.
            ground_truth: Optional[str]: Ground truth, expected output of the workflow.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            Trace: The created trace.
        """
        trace = Trace(
            input=input,
            output=output,
        )
        trace.add_llm_span(
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
        self.traces.append(trace)
        # Single span traces are automatically concluded so we reset the current parent.
        self.current_parent = None
        return trace

    def add_llm_span(
        self,
        input: LlmStepAllowedIOType,
        output: LlmStepAllowedIOType,
        model: str,
        tools: Optional[List[Dict]] = None,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        status_code: Optional[int] = None,
    ) -> LlmSpan:
        """
        Add a new llm span to the current parent.

        Parameters:
        ----------
            input: LlmStepAllowedIOType: Input to the node.
            output: LlmStepAllowedIOType: Output of the node.
            model: str: Model used for this span.
            tools: Optional[List[Dict]]: List of available tools passed to LLM on invocation.
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
        if self.current_parent is None:
            raise ValueError("A trace needs to be created in order to add a span.")
        span = self.current_parent.add_llm_span(
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
        )
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
    ) -> RetrieverSpan:
        """
        Add a new retriever span to the current parent.

        Parameters:
        ----------
            input: StepIOType: Input to the node.
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
        if self.current_parent is None:
            raise ValueError("A trace needs to be created in order to add a span.")
        span = self.current_parent.add_retriever_span(
            input=input,
            documents=documents,
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
            status_code=status_code,
        )
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
    ) -> ToolSpan:
        """
        Add a new tool span to the current parent.

        Parameters:
        ----------
            input: StepIOType: Input to the node.
            output: StepIOType: Output of the node.
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            ToolSpan: The created span.
        """
        if self.current_parent is None:
            raise ValueError("A trace needs to be created in order to add a span.")
        span = self.current_parent.add_tool_span(
            input=input,
            output=output,
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
            status_code=status_code,
        )
        return span

    def add_workflow_span(
        self,
        input: StepIOType,
        output: Optional[StepIOType] = None,
        name: Optional[str] = None,
        duration_ns: Optional[int] = None,
        created_at_ns: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> WorkflowSpan:
        """
        Add a workflow span to the current parent. This is useful when you want to create a nested workflow span
        within the trace or current workflow span. The next span you add will be a child of the current parent. To
        move out of the nested workflow, use conclude().

        Parameters:
        ----------
            input: StepIOType: Input to the node.
            output: Optional[StepIOType]: Output of the node. This can also be set on conclude().
            name: Optional[str]: Name of the span.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            created_at_ns: Optional[int]: Timestamp of the span's creation.
            metadata: Optional[Dict[str, str]]: Metadata associated with this span.
        Returns:
        -------
            WorkflowSpan: The created span.
        """
        if self.current_parent is None:
            raise ValueError("A trace needs to be created in order to add a span.")
        span = self.current_parent.add_workflow_span(
            input=input,
            output=output,
            name=name,
            duration_ns=duration_ns,
            created_at_ns=created_at_ns,
            metadata=metadata,
        )
        self.current_parent = span
        return span

    def conclude(
        self, output: Optional[StepIOType] = None, duration_ns: Optional[int] = None, status_code: Optional[int] = None
    ) -> Optional[StepWithChildSpans]:
        """
        Conclude the current trace or workflow span by setting the output of the current node. In the case of nested
        workflow spans, this will point the workflow back to the parent of the current workflow span.

        Parameters:
        ----------
            output: Optional[StepIOType]: Output of the node.
            duration_ns: Optional[int]: duration_ns of the node in nanoseconds.
            status_code: Optional[int]: Status code of the node execution.
        Returns:
        -------
            Optional[StepWithChildSpans]: The parent of the current workflow. None if no parent exists.
        """
        if self.current_parent is None:
            raise ValueError("No existing workflow to conclude.")
        self.current_parent = self.current_parent.conclude(
            output=output, duration_ns=duration_ns, status_code=status_code
        )
        return self.current_parent
