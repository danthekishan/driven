from dataclasses import dataclass, field
import time
from typing import Any, Awaitable, Callable, Literal, Optional, TypeAlias, Union


# ---------- Shared primitive types ----------
Role = Literal["system", "user", "assistant", "tool"]
ToolChoice = Literal["auto", "none", "required", "any", "tool_name"]

JSONValue: TypeAlias = Union[
    None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"], object
]
JSONSchema: TypeAlias = dict[str, Any]
Metadata: TypeAlias = dict[str, Any]
ProviderRaw: TypeAlias = dict[str, Any]
EventPayload: TypeAlias = dict[str, Any]
EventLogEntry: TypeAlias = dict[str, Any]


# ---------- Transcript/message model ----------
@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)


# ---------- Tool contracts ----------
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, JSONValue]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: JSONValue = None
    call_id: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Metadata = field(default_factory=dict)


# ---------- Usage ----------
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost: dict[str, float] = field(default_factory=dict)


# ---------- LLM I/O ----------
@dataclass
class LlmToolFunction:
    name: str
    description: str
    parameters_schema: JSONSchema = field(default_factory=dict)


@dataclass
class LlmInput:
    system: str
    messages: list[Message]
    tool_choice: Optional[ToolChoice] = None
    metadata: Metadata = field(default_factory=dict)


@dataclass
class LlmTextResponse:
    content: str
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: ProviderRaw = field(default_factory=dict)


@dataclass
class LlmTextDelta:
    text: str
    usage: Optional[Usage] = None


@dataclass
class LlmStructuredResponse:
    data: JSONValue
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: ProviderRaw = field(default_factory=dict)


@dataclass
class LlmOutput:
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw: ProviderRaw = field(default_factory=dict)


# ---------- Branch ----------
@dataclass(frozen=True)
class BranchInfo:
    branch_id: str
    parent_state_id: str
    parent_step: int
    spawned_at: float
    label: str = ""


# ---------- Trace ----------
@dataclass(frozen=True)
class TraceRecord:
    timestamp: float
    kind: str
    run_id: str
    step: int
    payload: EventPayload = field(default_factory=dict)


# ---------- Turn events emitted by controller ----------
@dataclass(frozen=True)
class ControllerRequested:
    message_count: int
    tool_count: int
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        return {
            "type": "controller_requested",
            "event": (
                f"Controller requested next turn with {self.message_count} messages "
                f"and {self.tool_count} available tools"
            ),
            "timestamp": self.timestamp,
            "context": {
                "message_count": self.message_count,
                "tool_count": self.tool_count,
            },
        }


@dataclass(frozen=True)
class LlmRequestPrepared:
    request: LlmInput
    tool_count: int
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        msg_count = len(self.request.messages)
        has_system = bool(self.request.system)
        return {
            "type": "llm_request_prepared",
            "event": (
                f"Prepared LLM request with {msg_count} messages"
                f" ({'with' if has_system else 'without'} system prompt)"
                f" and {self.tool_count} tools"
            ),
            "timestamp": self.timestamp,
            "context": {
                "message_count": msg_count,
                "has_system": has_system,
                "tool_count": self.tool_count,
            },
        }


@dataclass(frozen=True)
class LlmOutputReceived:
    output: LlmOutput
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        tool_names = [tc.name for tc in self.output.tool_calls]
        has_content = bool(self.output.content)
        return {
            "type": "llm_output_received",
            "event": (
                "Received LLM output: "
                f"{len(tool_names)} tool call(s), "
                f"content={'yes' if has_content else 'no'}, "
                f"stop_reason={self.output.stop_reason or 'unknown'}"
            ),
            "timestamp": self.timestamp,
            "context": {
                "tool_call_count": len(tool_names),
                "tool_names": tool_names,
                "has_content": has_content,
                "stop_reason": self.output.stop_reason,
            },
        }


@dataclass(frozen=True)
class ToolsRequested:
    tool_calls: list[ToolCall]
    raw_llm_response: ProviderRaw = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        tool_names = [tc.name for tc in self.tool_calls]
        return {
            "type": "tools_requested",
            "event": (
                f"Requested execution of {len(self.tool_calls)} tool call(s): "
                f"{', '.join(tool_names) if tool_names else 'none'}"
            ),
            "timestamp": self.timestamp,
            "context": {
                "tool_call_count": len(self.tool_calls),
                "tool_names": tool_names,
            },
        }


@dataclass(frozen=True)
class ToolsCompleted:
    results: list[ToolResult]
    raw_llm_response: ProviderRaw = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        success = sum(1 for r in self.results if r.ok)
        failed = len(self.results) - success
        failed_tools = [r.name for r in self.results if not r.ok]
        return {
            "type": "tools_completed",
            "event": (
                f"Tool execution completed: {success} succeeded, {failed} failed"
                + (f" (failed: {', '.join(failed_tools)})" if failed_tools else "")
            ),
            "timestamp": self.timestamp,
            "context": {
                "tool_result_count": len(self.results),
                "success": success,
                "failed": failed,
                "failed_tools": failed_tools,
            },
        }


@dataclass(frozen=True)
class AssistantFinalized:
    content: str
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    raw_llm_response: ProviderRaw = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        preview = self.content.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return {
            "type": "assistant_finalized",
            "event": (
                f"Assistant finalized response ({len(self.content)} chars, "
                f"stop_reason={self.stop_reason or 'unknown'}): {preview}"
            ),
            "timestamp": self.timestamp,
            "context": {
                "content_length": len(self.content),
                "stop_reason": self.stop_reason,
            },
        }


@dataclass(frozen=True)
class TurnFinished:
    done: bool
    reason: str
    timestamp: float = field(default_factory=time.time)

    def get_as_event(self) -> EventLogEntry:
        return {
            "type": "turn_finished",
            "event": f"Turn finished (done={self.done}, reason={self.reason})",
            "timestamp": self.timestamp,
            "context": {
                "done": self.done,
                "reason": self.reason,
            },
        }


TurnEvent = Union[
    ControllerRequested,
    LlmRequestPrepared,
    LlmOutputReceived,
    ToolsRequested,
    ToolsCompleted,
    AssistantFinalized,
    TurnFinished,
]

CallTools = Callable[[list[ToolCall], Optional[float]], Awaitable[list[ToolResult]]]


@dataclass
class RunOpts:
    llm: dict = field(default_factory=dict)

