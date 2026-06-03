from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    NamedTuple,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)

Role = Literal["system", "user", "assistant", "tool"]

JSONValue = Union[
    None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]
]
JSONSchema = dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    system: Optional[str] = None


# ---------- Actions decided by the controller ----------
@dataclass(frozen=True)
class RespondAction:
    content: str
    kind: Literal["say"] = field(default="say")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AskAction:
    question: str
    kind: Literal["ask"] = field(default="ask")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CallToolAction:
    tool_name: str
    kind: Literal["call_tool"] = field(default="call_tool")
    arguments: dict[str, JSONValue] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinishAction:
    content: Optional[str]
    kind: Literal["finish"] = field(default="finish")
    reason: str = field(default="done")
    metadata: dict[str, Any] = field(default_factory=dict)


Action = Union[RespondAction, AskAction, CallToolAction, FinishAction]


# ---------- Observations (results surfaced back into state) ----------
@dataclass(frozen=True)
class Observation:
    data: Any
    kind: Literal["tool_result", "ask_answer", "error"]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------- Events for realtime emit ----------

# ==========================================
# 1. Define the NamedTuple Schemas
# ==========================================


class EventType(Enum):
    LLM_STARTED = auto()
    LLM_ERROR = auto()
    LLM_ENDED = auto()
    LLM_USAGE = auto()
    LLM_THINK = auto()

    TOOL_CALL_STARTED = auto()
    TOOL_CALL_ERROR = auto()
    TOOL_CALL_ENDED = auto()

    ACTION_DECIDED = auto()
    OBSERVATION_RECEIVED = auto()

    STEP_STARTED = auto()


class LlmRequestEventPayload(NamedTuple):
    type: Literal[EventType.LLM_STARTED, EventType.TOOL_CALL_STARTED]
    op_type: Literal["tool", "generate"]
    model: str
    prompt: str
    extra: dict


class LlmResponseEventPayload(NamedTuple):
    type: Literal[EventType.LLM_THINK, EventType.LLM_ENDED, EventType.TOOL_CALL_ENDED]
    model: str
    prompt: str
    op_type: Literal["tool", "generate", "think"]
    response: dict[str, Any]
    extra: dict


class LlmErrorEventPayload(NamedTuple):
    model: str
    error_message: str
    status_code: int
    op_type: Literal["tool", "generate"]
    type: Literal[EventType.LLM_ERROR, EventType.TOOL_CALL_ERROR]
    extra: dict


class UsageEventPayload(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    type: Literal[EventType.LLM_USAGE] = EventType.LLM_USAGE


class ActionEventPayload(NamedTuple):
    action: Action
    agent_messages: list[Message]
    type: Literal[EventType.ACTION_DECIDED] = EventType.ACTION_DECIDED


class ObservationEventPayload(NamedTuple):
    observation: Observation
    tool_messages: list[Message]
    type: Literal[EventType.OBSERVATION_RECEIVED] = EventType.OBSERVATION_RECEIVED


class GenericEventPayload(NamedTuple):
    type: Literal[EventType.STEP_STARTED]
    data: dict[str, Any]


EventTypesPayload = Union[
    LlmRequestEventPayload,
    LlmResponseEventPayload,
    LlmErrorEventPayload,
    UsageEventPayload,
    ActionEventPayload,
    ObservationEventPayload,
    GenericEventPayload,
]


@dataclass
class AgentEvent:
    payload: EventTypesPayload

    @property
    def type(self) -> EventType:
        return self.payload.type


# ---------- Llm ----------
@dataclass
class LlmToolFunction:
    """Represents a tool/function exposed to an Llm."""

    name: str
    description: str
    parameters_schema: JSONSchema = field(default_factory=dict)


ToolChoice = Literal["auto", "none", "required", "any", "tool_name"]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, JSONValue] = field(default_factory=dict)
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    ok: bool
    content: JSONValue = None
    call_id: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost: dict[str, float] = field(default_factory=dict)


@dataclass
class LlmInput:
    input: list[Message]
    history: list[Message]
    tool_choice: Optional[ToolChoice] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Non-streaming text
@dataclass
class LlmTextResponse:
    content: str
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: dict[str, Any] = field(default_factory=dict)


# Streaming text delta
@dataclass
class LlmTextDelta:
    text: str
    usage: Optional[Usage] = None


# Structured output (non-streaming)
@dataclass
class LlmStructuredResponse:
    data: JSONValue
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: dict[str, Any] = field(default_factory=dict)


# Tool calling (non-streaming) one-of
@dataclass
class LlmOutput:
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    stop_reason: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)


# Streaming tool call events
@dataclass
class LlmToolCallStarted:
    call_id: str
    name: str
    args_partial: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmToolCallDelta:
    call_id: str
    args_partial: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmToolCallFinished:
    call_id: str
    name: str
    args_final: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    id: str
    state_id: str
    get_tools_info: Callable[..., list[LlmToolFunction]]
    tool_choice: ToolChoice = field(default="auto")
    history: list[Message] = field(default_factory=list)
    input: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    state_id: str
    run_context: RunContext
    step: int = 0
    history: list[Message] = field(default_factory=list)
    input: list[Message] = field(default_factory=list)
    last_action: Optional[Action] = field(default=None)
    last_observation: Optional[Observation] = field(default=None)
    done: bool = field(default=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_run_context(
        self, run_id: str, get_tools_info: Callable[..., list[LlmToolFunction]]
    ) -> RunContext:
        return RunContext(
            id=run_id,
            state_id=self.state_id,
            get_tools_info=get_tools_info,
            history=self.history,
            input=self.input,
            metadata=self.metadata,
        )
