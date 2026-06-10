from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)

from driven.core.schemas import (
    BranchInfo,
    EventLogEntry,
    LlmInput,
    LlmOutput,
    LlmStructuredResponse,
    LlmTextDelta,
    LlmTextResponse,
    LlmToolFunction,
    Message,
    ToolCall,
    ToolResult,
    TraceRecord,
    TurnEvent,
    CallTools,
)


@dataclass
class HarnessState:
    state_id: str
    system: str = ""
    prompt: str = ""
    step: int = field(default=0)
    messages: list[Message] = field(default_factory=list)
    event_log: list[EventLogEntry] = field(default_factory=list)
    done: bool = False
    internal: dict[str, Any] = field(default_factory=dict)
    branch: Optional[BranchInfo] = None
    branches: list[BranchInfo] = field(default_factory=list)


@runtime_checkable
class Emitter(Protocol):
    async def __call__(self, event: dict[str, Any]) -> None: ...


@runtime_checkable
class Llm(Protocol):
    async def generate_text(self, request: LlmInput) -> LlmTextResponse: ...
    async def generate_structured(
        self, request: LlmInput, schema: dict
    ) -> LlmStructuredResponse: ...
    async def chat_with_tools(
        self, request: LlmInput, tools: list[LlmToolFunction]
    ) -> LlmOutput: ...
    async def generate_text_stream(
        self, request: LlmInput
    ) -> AsyncIterator[LlmTextDelta]: ...


@runtime_checkable
class Runtime(Protocol):
    async def connect(self, create_branch: Callable) -> Any: ...
    async def disconnect(self): ...
    def get_tools(self, *args, **kwargs) -> list[LlmToolFunction]: ...
    async def call_tools(
        self,
        tools: list[ToolCall],
        state: HarnessState,
        llm: Llm,
        emitter: Optional[Emitter] = None,
        timeout: Optional[float] = None,
    ) -> list[ToolResult]: ...


@runtime_checkable
class Controller(Protocol):
    def __call__(
        self,
        messages: list[Message],
        system: str,
        get_tools: Callable[..., list[LlmToolFunction]],
        call_tools: CallTools,
        llm: Llm,
    ) -> AsyncIterator[TurnEvent]: ...


@runtime_checkable
class StateManager(Protocol):
    async def load(self, run_id: str) -> HarnessState: ...
    async def save(self, state: HarnessState) -> HarnessState: ...


@runtime_checkable
class TraceSink(Protocol):
    async def __call__(self, trace: TraceRecord) -> None: ...


@dataclass
class HarnessRuntime:
    state_manager: StateManager
    controller: Controller
    runtime: Runtime
    llm: Llm
    emitter: Optional[Emitter] = None
    trace_sink: Optional[TraceSink] = None
    private_of: Optional[str] = None
    _connected: bool = field(default=False, init=False)

    def _require_connected(self):
        if not self._connected:
            raise RuntimeError(
                "runtime not connected — enter harness as context manager first"
            )

    async def call_tools(
        self,
        state: HarnessState,
        tool_calls: list[ToolCall],
        timeout: Optional[float] = None,
    ) -> list[ToolResult]:
        return await self.runtime.call_tools(
            tools=tool_calls,
            state=state,
            llm=self.llm,
            emitter=self.emitter,
            timeout=timeout,
        )

    def get_tools(
        self,
    ) -> list[LlmToolFunction]:
        return self.runtime.get_tools(private_of=self.private_of)

    async def run_controller(self, state: HarnessState):
        async def _call_tools(
            tool_calls: list[ToolCall], timeout: Optional[float] = None
        ):
            return await self.call_tools(state, tool_calls, timeout)

        async for event in self.controller(
            messages=state.messages,
            system=state.system,
            get_tools=self.get_tools,
            call_tools=_call_tools,
            llm=self.llm,
        ):
            yield event
