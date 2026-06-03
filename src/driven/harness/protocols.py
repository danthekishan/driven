from typing import (
    AsyncIterator,
    Optional,
    Protocol,
    runtime_checkable,
    Callable,
    Union,
    Sequence,
)

from driven.harness.types import (
    LlmToolFunction,
    Message,
    AgentState,
    LlmInput,
    LlmTextDelta,
    LlmStructuredResponse,
    LlmTextResponse,
    LlmOutput,
    ActionEvent,
    ObservationEvent,
    CallToolAction,
    AnyEvent,
)


@runtime_checkable
class Emitter(Protocol):
    async def emit(self, events: Union[AnyEvent, Sequence[AnyEvent]]) -> None: ...


@runtime_checkable
class StateManager(Protocol):
    async def load(self, run_id: str, input: list[Message]) -> AgentState: ...
    async def save(self, state: AgentState) -> AgentState: ...


@runtime_checkable
class Reducer(Protocol):
    async def apply(
        self,
        state: AgentState,
        events: Union[ObservationEvent, Sequence[ObservationEvent]],
    ) -> AgentState: ...


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
    def get_tools(self, *args, **kwargs) -> list[LlmToolFunction]: ...
    async def execute(
        self,
        action: CallToolAction,
        state: AgentState,
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> list[ObservationEvent]: ...


@runtime_checkable
class Controller(Protocol):
    """Decides the next action(s) given the current state."""

    async def decide(
        self,
        messages: list[Message],
        get_tools: Callable[..., list[LlmToolFunction]],
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> list[ActionEvent]: ...
