import traceback
from collections import defaultdict
from json import dumps, loads
from time import time_ns
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import __all__ as LangchainEmbeddingNames
from langchain.prompts.chat import ChatPromptValue
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.document import Document
from langchain.schema.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain.schema.messages import _message_to_dict as message_to_dict
from langchain.schema.output import ChatGeneration, ChatGenerationChunk, LLMResult
from langchain.vectorstores import __all__ as LangchainVectorStoreNames
from pydantic import BaseModel
from pydantic.v1 import BaseModel as BaseModelV1

from galileo_core.schemas.shared.workflows.node_type import NodeType
from promptquality.chain_run_module import chain_run
from promptquality.constants.run import TagType
from promptquality.constants.scorers import Scorers
from promptquality.set_config_module import set_config
from promptquality.types.chains.row import NodeRow
from promptquality.types.config import Config
from promptquality.types.custom_scorer import CustomScorer
from promptquality.types.customized_scorer import CustomizedChainPollScorer
from promptquality.types.registered_scorers import RegisteredScorer
from promptquality.types.run import RunTag, ScorersConfiguration
from promptquality.utils.logger import logger

Stringable = (str, int, float)
EmbeddingRunTagKey = "Embedding Model"
VectorStoreRunTagKey = "Vector Store"


class GalileoPromptCallback(BaseCallbackHandler):
    def __init__(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        scorers: Optional[List[Union[Scorers, CustomizedChainPollScorer, CustomScorer, RegisteredScorer, str]]] = None,
        generated_scorers: Optional[List[str]] = None,
        run_tags: Optional[List[RunTag]] = None,
        scorers_config: ScorersConfiguration = ScorersConfiguration(),
        wait: bool = True,
        config: Optional[Config] = None,
        **kwargs: Any,
    ) -> None:
        """
        LangChain callbackbander for logging prompts to Galileo.

        Parameters
        ----------
        project_name : str
            Name of the project to log to
        """
        # Galileo parameters.
        self.project_name = project_name
        self.run_name = run_name
        self.scorers = scorers
        self.scorers_config = scorers_config
        # Force generated_scorers to be a list due to Pydantic evaluating plain strings as lists.
        if isinstance(generated_scorers, str):
            generated_scorers = [generated_scorers]
        self.generated_scorers = generated_scorers
        self.run_tags = run_tags
        self.wait = wait
        self.config = config or set_config()

        # Row information.
        # Mapping of root of the chain to all children.
        self.root_child_mapping: DefaultDict = defaultdict(list)
        self.root_nodes: List[UUID] = list()
        # Mapping of parent to children.
        self.parent_child_mapping: DefaultDict = defaultdict(list)
        # Mapping of child to parent.
        self.child_parent_mapping: Dict[UUID, Optional[UUID]] = dict()

        self.chain_inputs: Dict[UUID, Dict[str, Any]] = dict()
        self.serializations: Dict[UUID, Dict[str, Any]] = dict()

        self.rows: Dict[UUID, NodeRow] = dict()

    def set_relationships(self, run_id: UUID, node_type: NodeType, parent_run_id: Optional[UUID] = None) -> None:
        self.child_parent_mapping[run_id] = parent_run_id
        if parent_run_id:
            self.parent_child_mapping[parent_run_id].append(run_id)
            self.rows[parent_run_id].has_children = True
        root_id = self.get_root_id(run_id)
        self.root_child_mapping[root_id].append(run_id)
        self.rows[run_id] = NodeRow(
            node_id=run_id,
            node_type=node_type,
            # -1 because the step is incremented after the callback is run.
            step=len(self.root_child_mapping[root_id]) - 1,
            chain_id=parent_run_id,
            chain_root_id=root_id,
        )
        if root_id == run_id:
            self.root_nodes.append(run_id)

    def get_root_id(self, run_id: UUID) -> UUID:
        parent_id = self.child_parent_mapping[run_id]
        if parent_id:
            return self.get_root_id(parent_id)
        else:
            return run_id

    def mark_step_start(
        self,
        run_id: UUID,
        node_name: str,
        serialized: Optional[Dict[str, Any]],
        prompt: Optional[str] = None,
        node_input: str = "",
        **kwargs: Dict[str, Any],
    ) -> None:
        root_id = self.get_root_id(run_id)
        self.rows[run_id].prompt = prompt
        self.rows[run_id].node_input = node_input
        self.rows[run_id].params = kwargs.get("invocation_params", dict())
        self.rows[run_id].inputs = self.chain_inputs.get(root_id, dict())
        tools = self.rows[run_id].params.get("tools", None)
        self.rows[run_id].tools = dumps(tools) if tools else None
        # Parse and set the name of the node.
        node_class_reference = None
        if serialized is not None and isinstance(serialized, dict):
            node_name = serialized.get("name") or node_name
            node_class_reference = serialized.get("id")
        if node_class_reference and isinstance(node_class_reference, list):
            self.rows[run_id].node_name = node_class_reference[-1]
        else:
            self.rows[run_id].node_name = node_name

    def mark_step_end(
        self, run_id: UUID, response: Optional[str] = None, node_output: str = "", **kwargs: Dict[str, Any]
    ) -> None:
        self.rows[run_id].latency = time_ns() - self.rows[run_id].creation_timestamp
        if response:
            self.rows[run_id].response = response
        # Agents set node_output before the chain_end step, so check if value already set
        if not self.rows[run_id].node_output:
            self.rows[run_id].node_output = node_output

    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever starts running."""
        self.set_relationships(run_id, node_type=NodeType.retriever, parent_run_id=parent_run_id)
        self.mark_step_start(run_id, node_name="Retriever", serialized=serialized, node_input=query, **kwargs)
        # Try to add relevant tags to self.run_tags
        existing_tag_keys = [tag.key for tag in self.run_tags] if self.run_tags else []
        if tags:
            run_tags = self.run_tags or []
            for tag in tags:
                if tag in LangchainEmbeddingNames and EmbeddingRunTagKey not in existing_tag_keys:
                    run_tags.append(RunTag(key=EmbeddingRunTagKey, value=tag, tag_type=TagType.RAG))
                if tag in LangchainVectorStoreNames and VectorStoreRunTagKey not in existing_tag_keys:
                    run_tags.append(RunTag(key=VectorStoreRunTagKey, value=tag, tag_type=TagType.RAG))
            self.run_tags = run_tags

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running."""
        node_output = dumps(
            [{"page_content": document.page_content, "metadata": document.metadata} for document in documents]
        )
        self.mark_step_end(run_id, node_output=node_output, **kwargs)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors."""
        del self.rows[run_id]

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool starts running."""
        self.set_relationships(run_id, node_type=NodeType.tool, parent_run_id=parent_run_id)
        self.mark_step_start(run_id, node_name="Tool", serialized=serialized, node_input=input_str, **kwargs)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running."""
        self.mark_step_end(
            run_id,
            node_output=dumps(
                output,
                default=GalileoPromptCallback.json_serializer,
            ),
            **kwargs,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors."""
        error_text = repr(error)
        if isinstance(error, BaseException):
            error_text += "\n" + "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.mark_step_end(
            run_id,
            node_output=f"ERROR: {error_text}",
            **kwargs,
        )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Run on agent finish.

        The order of operations for agents is on_chain_start, on_agent_action x times, on_agent_finish, on_chain_finish.
        We are creating the agent node with on_chain_start, then populating all of it's agent specific data in
        on_agent_finish. We are skipping on_agent_action, because there is no relevant info there as of yet and it could
        also be called 0 times.
        """
        self.rows[run_id].node_type = NodeType.agent
        self.rows[run_id].node_input = self.chain_inputs[run_id].get("input", "")
        self.rows[run_id].node_output = finish.return_values.get("output") or finish.log

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        self.set_relationships(run_id, node_type=NodeType.llm, parent_run_id=parent_run_id)
        prompt = prompts[0]
        self.mark_step_start(run_id, node_name="LLM", serialized=serialized, prompt=prompt, node_input=prompt, **kwargs)

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        self.set_relationships(run_id, node_type=NodeType.chat, parent_run_id=parent_run_id)

        chat_messages = ChatPromptValue(messages=messages[0])
        prompt = chat_messages.to_string()
        self.mark_step_start(
            run_id, node_name="Chat", serialized=serialized, prompt=prompt, node_input=prompt, **kwargs
        )

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when LLM ends running."""

        # Flatten the response and pull out the generation.
        generation = response.flatten()[0].generations[0][0]
        tool_calls = None
        # Only chat generations have messages which contain the invocation params
        if isinstance(generation, (ChatGeneration, ChatGenerationChunk)):
            message = generation.message
            if isinstance(message, (AIMessageChunk, AIMessage)):
                tool_calls = message.tool_calls
        output_text = generation.text
        if not output_text and tool_calls:
            output_text = dumps(tool_calls)
        self.mark_step_end(run_id, response=output_text, node_output=output_text, **kwargs)

        usage_data = dict()
        # OpenAI/Anthropic format.
        if response.llm_output:
            # OpenAI format.
            usage_data = response.llm_output.get("token_usage", response.llm_output.get("usage", {}))
        elif generation.generation_info:
            # Vertex AI format.
            usage_data = generation.generation_info.get("usage_metadata", dict())

        if usage_data:
            self.rows[run_id].query_input_tokens = (
                usage_data.get("prompt_tokens")
                or usage_data.get("prompt_token_count")
                or usage_data.get("input_tokens")
                or 0
            )
            self.rows[run_id].query_output_tokens = (
                usage_data.get("completion_tokens")
                or usage_data.get("candidates_token_count")
                or usage_data.get("output_tokens")
                or 0
            )
            self.rows[run_id].query_total_tokens = (
                usage_data.get("total_tokens") or usage_data.get("total_token_count") or 0
            )

        if generation.generation_info:
            self.rows[run_id].finish_reason = generation.generation_info.get("finish_reason", "")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors."""
        del self.rows[run_id]

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Union[Dict[str, Any], Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run when chain starts running.

        The inputs here are expected to only be a dictionary per the `langchain` docs
        but from experience, we do see strings and `BaseMessage`s in there, so we
        support those as well.
        """
        self.set_relationships(run_id, node_type=NodeType.chain, parent_run_id=parent_run_id)
        if isinstance(inputs, Stringable):
            node_input: Union[Dict, List] = {"input": inputs}
        elif isinstance(inputs, BaseMessage):
            node_input = message_to_dict(inputs)
        elif isinstance(inputs, dict):
            node_input = dict()
            for key, value in inputs.items():
                if isinstance(value, Stringable):
                    node_input[key] = value
                elif isinstance(value, BaseMessage):
                    node_input[key] = str(message_to_dict(value))
                elif isinstance(value, Sequence):
                    node_input[key] = ". ".join(repr(v) for v in value)
        elif isinstance(inputs, list) and all(isinstance(v, Document) for v in inputs):
            node_input = [{"page_content": document.page_content, "metadata": document.metadata} for document in inputs]
        # Inputs can be an Agent object on some chains
        elif isinstance(inputs, (AgentFinish, AgentAction)):
            # Serialize agent object to dict and get the stringable values.
            node_input = {key: value for key, value in inputs.dict().items() if value and isinstance(value, Stringable)}
        else:
            logger.warning(f"Unsupported input type {type(inputs)} for {run_id=}, {parent_run_id=}. Skipping...")
            node_input = dict()
        self.chain_inputs[run_id] = (
            node_input if isinstance(node_input, Dict) else {str(i): val for i, val in enumerate(node_input)}
        )
        self.mark_step_start(run_id, node_name="Chain", serialized=serialized, node_input=dumps(node_input), **kwargs)

    def on_chain_end(
        self, outputs: Union[str, Dict[str, Any]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when chain ends running."""
        self.rows[run_id].finish_reason = "chain_end"
        response: Optional[str] = None

        if isinstance(outputs, str):
            node_output = {"output": outputs}
        elif isinstance(outputs, dict):
            # casting to a str because it should work somewhat cleanly with most output parser types
            # https://python.langchain.com/docs/modules/model_io/output_parsers/types
            # The 'output' key is present on agent chains
            response = str(outputs.get("text", "")) or str(outputs.get("output", ""))
            node_output = outputs
        elif isinstance(outputs, BaseModel):
            node_output = outputs.model_dump()
        elif isinstance(outputs, BaseModelV1):
            node_output = outputs.dict()
        elif isinstance(outputs, BaseMessage):
            node_output = message_to_dict(outputs)
        elif isinstance(outputs, Sequence):
            node_output = [
                message_to_dict(output) if isinstance(output, BaseMessage) else repr(output) for output in outputs
            ]
        else:
            logger.warning(f"Unsupported output type {type(outputs)} for {run_id=}, {parent_run_id=}. Skipping...")
            node_output = dict()

        self.mark_step_end(
            run_id,
            response=response,
            node_output=dumps(
                node_output,
                default=GalileoPromptCallback.json_serializer,
            ),
            **kwargs,
        )

    def on_chain_error(
        self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    @staticmethod
    def json_serializer(obj: Any) -> Union[str, Dict[Any, Any]]:
        """
        For serializing objects that cannot be serialized by default with json.dumps.

        Checks for certain methods to convert object to dict.
        """
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, BaseModelV1):
            return loads(obj.json())
        logger.warning(f"Unable to serialize {type(obj)}, for visualization in Galileo console")
        return str(type(obj))

    def add_targets(self, targets: List[str]) -> None:
        """
        targets: List[str]:
            A list of target outputs. The list should be the length of the number of chain invokations.
            Targets will be mapped to chain root nodes.
        """
        if len(targets) != len(self.root_nodes):
            raise ValueError(
                f"Got {len(targets)} targets but expected {len(self.root_nodes)}"
                "The number of targets must match the number of chain invokations."
            )
        for root_id, target in zip(self.root_nodes, targets):
            self.rows[root_id].target = target

    def finish(self) -> None:
        rows_to_log: List[NodeRow] = list()
        # Exclude chains that don't have children.
        for root_id in self.root_nodes:
            for node_id in self.root_child_mapping[root_id]:
                row = self.rows.get(node_id)
                if row is None:
                    logger.debug(f"Skipping row {node_id} because its row doesn't exist.")
                # If the node is a non-chain node, or is a parent node with any children, include it.
                elif row.node_type != NodeType.chain or node_id in self.parent_child_mapping.keys():
                    rows_to_log.append(row)
                else:
                    logger.debug(f"Skipping row {node_id} because it is an empty chain.")
        chain_run(
            rows_to_log,
            self.project_name,
            self.run_name,
            self.scorers,
            generated_scorers=self.generated_scorers,
            scorers_config=self.scorers_config,
            run_tags=self.run_tags,
            wait=self.wait,
            config=self.config,
        )
