from functools import wraps
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)
from uuid import uuid4

from driven.core.schemas import (
    AnyEvent,
    AssistantResp,
    JSONValue,
    Message,
    ActionEvent,
    ObservationEvent,
    CallToolAction,
    LlmToolFunction,
    LlmInput,
    LlmTextDelta,
    LlmStructuredResponse,
    LlmTextResponse,
    LlmOutput,
    RespondAction,
    ToolFailed,
    ToolProduced,
)


@dataclass
class HarnessState:
    state_id: str
    prompt: str
    step: int = field(default=0)
    messages: list[Message] = field(default_factory=list)
    # per-step
    recent_actions: list[ActionEvent] = field(default_factory=list)
    recent_observations: list[ObservationEvent] = field(default_factory=list)
    actions_log: list[list[ActionEvent]] = field(default_factory=list)
    observations_log: list[list[ObservationEvent]] = field(default_factory=list)
    # Control
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Emitter(Protocol):
    async def emit(
        self,
        event_type: str,
        events: Union[dict, AnyEvent, Sequence[dict], Sequence[AnyEvent]],
    ) -> None: ...


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
    async def __aenter__(self) -> Any: ...
    async def __aexit__(self, exc_type, exc, tb): ...
    def get_tools(self, *args, **kwargs) -> list[LlmToolFunction]: ...
    async def call_tool(
        self,
        fn_name: str,
        arguments: dict[str, JSONValue],
        state: HarnessState,
        llm: Llm,
        emitter: Optional[Emitter] = None,
        timeout: Optional[float] = None,
    ) -> list[ObservationEvent]: ...


@runtime_checkable
class Controller(Protocol):
    """Decides the next action(s) given the current state."""

    async def decide(
        self,
        messages: list[Message],
        get_tools: Callable[..., list[LlmToolFunction]],
        llm: Llm,
        emitter: Optional[Emitter] = None,
    ) -> list[ActionEvent]: ...


@runtime_checkable
class StateManager(Protocol):
    async def load(self, run_id: str) -> HarnessState: ...
    async def save(self, state: HarnessState) -> HarnessState: ...


@dataclass
class HarnessContext:
    state_manager: StateManager
    controller: Controller
    runtime: Runtime
    llm: Llm
    emitter: Optional[Emitter] = None


async def _handle_action(
    state: HarnessState, ctx: HarnessContext, action: ActionEvent
) -> list[ObservationEvent]:
    match action:
        case RespondAction(content):
            return [AssistantResp(content)]
        case CallToolAction(name, arguments, timeout):
            return await ctx.runtime.call_tool(
                name, arguments, state, ctx.llm, ctx.emitter, timeout
            )
        case _:
            raise RuntimeError(f"unknown action type: {type(action)}")


async def _handle_observation(
    observation: ObservationEvent,
) -> tuple[list[Message], bool]:
    match observation:
        case AssistantResp(content):
            msgs = [Message(role="assistant", content=content)]
            return (msgs, True)
        case ToolProduced(tool_name, content):
            msgs = [
                Message(
                    role="tool",
                    name=tool_name,
                    content=str(content),
                )
            ]
            return (msgs, False)

        case ToolFailed(tool_name, error_type, message):
            msgs = [
                Message(
                    role="tool",
                    name=tool_name,
                    content=f"[{error_type}] {message}",
                )
            ]
            return (msgs, False)

        case _:
            raise RuntimeError(f"unknown observation type: {type(observation)}")


async def _run(state: HarnessState, ctx: HarnessContext) -> HarnessState:
    state.step += 1  # advance step

    actions = await ctx.controller.decide(
        messages=state.messages,
        get_tools=ctx.runtime.get_tools,
        llm=ctx.llm,
        emitter=ctx.emitter,
    )

    observations: list[ObservationEvent] = []

    if ctx.emitter and actions:
        await ctx.emitter.emit("actions", actions)

    for action in actions:
        _obs = await _handle_action(state, ctx, action)
        observations.extend(_obs)

    if ctx.emitter and observations:
        await ctx.emitter.emit("observations", observations)

    # update state
    state.recent_actions = actions
    state.recent_observations = observations
    state.actions_log.append(actions)
    state.observations_log.append(observations)

    for event in observations:
        messages, completed = await _handle_observation(event)
        state.messages.extend(messages)

        if completed:
            state.done = True

    return state


type Next = Callable[[HarnessState, HarnessContext], Awaitable[HarnessState]]


class Middleware(Protocol):
    async def __call__(
        self, state: HarnessState, ctx: HarnessContext, next: Next
    ) -> HarnessState: ...


type Middlewares = list[Middleware]


def wrap_middleware(middlware: Middleware, next: Next) -> Next:
    @wraps(next)
    async def wrapped(state: HarnessState, ctx: HarnessContext) -> HarnessState:
        return await middlware(state, ctx, next)

    return wrapped


def build_chain(middlewares: Middlewares, handler: Next) -> Next:
    chain = handler

    for mw in reversed(middlewares):
        chain = wrap_middleware(mw, chain)

    return chain


class Harness:
    def __init__(
        self,
        ctx: HarnessContext,
        step_middlewares: Middlewares,
        session_middlewares: Middlewares,
    ):
        self.ctx = ctx
        self.step_runner = build_chain(step_middlewares, _run)
        self.session_runner = build_chain(session_middlewares, self._run_loop)

    async def __aenter__(self):
        await self.ctx.runtime.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.ctx.runtime.__aexit__(exc_type, exc, tb)

    async def _run_loop(self, state: HarnessState, ctx: HarnessContext):
        while not state.done:
            state = await self.step_runner(state, ctx)

        return state

    async def run(
        self,
        run_id: Optional[str] = None,
        prompt: str = "",
        input: list[Message] | None = None,
    ):
        rid = str(run_id or uuid4())
        state: Optional[HarnessState] = None
        try:
            state = await self.ctx.state_manager.load(rid)

            # Inject initial prompt and input if this looks like a new state
            if prompt:
                state.prompt = prompt
                state.messages.append(Message(role="user", content=prompt))
            elif input:
                state.messages.extend(input)
            else:
                raise ValueError("Either prompt or input must be provided")

            state = await self.session_runner(state, self.ctx)
            return state
        finally:
            if state is not None:
                await self.ctx.state_manager.save(state)
