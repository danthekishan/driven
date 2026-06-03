from typing import (
    AsyncIterator,
    Optional,
    Protocol,
    runtime_checkable,
)

from driven.harness.types import (
    LlmToolFunction,
    Message,
    Observation,
    RunContext,
    AgentState,
    AgentEvent,
    LlmInput,
    LlmTextDelta,
    LlmStructuredResponse,
    LlmTextResponse,
    LlmOutput,
    Action,
)


@runtime_checkable
class Emitter(Protocol):
    async def emit(self, event: AgentEvent) -> None: ...


@runtime_checkable
class StateManager(Protocol):
    async def load(self, run_id: str, input: list[Message]) -> AgentState: ...
    async def save(self, state: AgentState) -> AgentState: ...


@runtime_checkable
class Reducer(Protocol):
    async def apply(self, state: AgentState, event: AgentEvent) -> AgentState: ...


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
        action: Action,
        state: AgentState,
        run_context: RunContext,
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> tuple[Observation, list[Message]]: ...


@runtime_checkable
class Controller(Protocol):
    """Decides the next action given the current state.

    Implementations may wrap an LLM (LLMController), a scripted list (ScriptedController),
    or a human-in-the-loop controller. The harness does not construct tool lists or schemas;
    that is up to the Controller implementation.
    """

    async def decide(
        self,
        run_context: RunContext,
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> tuple[Action, list[Message]]: ...
