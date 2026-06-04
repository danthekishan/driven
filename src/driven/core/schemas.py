from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

Role = Literal["system", "user", "assistant", "tool"]

JSONValue = Union[
    None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"], object
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


# ---------- ActionEvents (from Controller) ----------
@dataclass(frozen=True)
class RespondAction:
    content: str


@dataclass(frozen=True)
class CallToolAction:
    name: str
    arguments: dict[str, JSONValue] = field(default_factory=dict)
    timeout: Optional[float] = None


ActionEvent = Union[RespondAction, CallToolAction]


# ---------- ObservationEvents (to Reducer) ----------
@dataclass(frozen=True)
class AssistantResp:
    content: str


@dataclass(frozen=True)
class ToolProduced:
    tool_name: str
    content: JSONValue


@dataclass(frozen=True)
class ToolFailed:
    tool_name: str
    error_type: str
    message: str


ObservationEvent = Union[AssistantResp, ToolProduced, ToolFailed]
AnyEvent = Union[ActionEvent, ObservationEvent]


# ---------- LLM contracts ----------
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
    messages: list[Message]
    tool_choice: Optional[ToolChoice] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmTextResponse:
    content: str
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmTextDelta:
    text: str
    usage: Optional[Usage] = None


@dataclass
class LlmStructuredResponse:
    data: JSONValue
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LlmOutput:
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    stop_reason: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)
