from functools import wraps
from dataclasses import asdict, dataclass, field
import time
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    runtime_checkable,
)
from uuid import uuid4

from driven.core.schemas import (
    AssistantFinalized,
    BranchInfo,
    ControllerRequested,
    EventLogEntry,
    LlmInput,
    LlmOutput,
    LlmOutputReceived,
    LlmRequestPrepared,
    LlmStructuredResponse,
    LlmTextDelta,
    LlmTextResponse,
    LlmToolFunction,
    Message,
    ToolCall,
    ToolResult,
    ToolsCompleted,
    ToolsRequested,
    TraceRecord,
    TurnEvent,
    TurnFinished,
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


SpawnBranch = Callable[
    ...,
    Awaitable[tuple[Optional[HarnessState], Optional[Exception]]],
]

CallTools = Callable[[list[ToolCall], Optional[float]], Awaitable[list[ToolResult]]]


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
    async def connect(self) -> Any: ...
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
class HarnessContext:
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


async def _emit(
    ctx: HarnessContext, state: HarnessState, name: str, payload: dict[str, Any]
):
    if not ctx.emitter:
        return
    event: dict[str, Any] = {
        "source": "harness",
        "name": name,
        "timestamp": time.time(),
        "run_id": state.state_id,
        "step": state.step,
        "payload": payload,
    }
    if state.branch:
        event["branch"] = {
            "branch_id": state.branch.branch_id,
            "parent_run_id": state.branch.parent_run_id,
            "parent_step": state.branch.parent_step,
            "label": state.branch.label,
        }
    await ctx.emitter(event)


def _messages_from_event(event: TurnEvent) -> tuple[list[Message], bool]:
    match event:
        case AssistantFinalized(content=content):
            return [Message(role="assistant", content=content)], True

        case ToolsRequested(tool_calls=tool_calls):
            return [
                Message(
                    role="assistant",
                    content="",
                    metadata={
                        "tool_calls": [
                            {
                                "id": tc.call_id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in tool_calls
                        ]
                    },
                )
            ], False

        case ToolsCompleted(results=results):
            out: list[Message] = []
            for res in results:
                if res.ok:
                    content = str(res.content)
                else:
                    content = f"[{res.error_type}] {res.error_message}"
                out.append(
                    Message(
                        role="tool",
                        name=res.name,
                        tool_call_id=res.call_id,
                        content=content,
                    )
                )
            return out, False

        case TurnFinished(done=done):
            return [], done

        case ControllerRequested() | LlmRequestPrepared() | LlmOutputReceived():
            return [], False

        case _:
            return [], False


def _trace_records_from_event(
    run_id: str, step: int, event: TurnEvent
) -> list[TraceRecord]:
    match event:
        case LlmRequestPrepared(request=request, timestamp=timestamp):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="llm_request",
                    run_id=run_id,
                    step=step,
                    payload=asdict(request),
                )
            ]

        case LlmOutputReceived(output=output, timestamp=timestamp):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="llm_response",
                    run_id=run_id,
                    step=step,
                    payload={
                        "stop_reason": output.stop_reason,
                        "tool_call_count": len(output.tool_calls),
                        "content": output.content,
                        "usage": asdict(output.usage) if output.usage else None,
                        "raw": output.raw,
                    },
                )
            ]

        case ToolsRequested(
            tool_calls=tool_calls, raw_llm_response=raw, timestamp=timestamp
        ):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="tools_requested",
                    run_id=run_id,
                    step=step,
                    payload={
                        "tool_calls": [asdict(tc) for tc in tool_calls],
                        "raw_llm_response": raw,
                    },
                )
            ]

        case ToolsCompleted(results=results, raw_llm_response=raw, timestamp=timestamp):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="tools_completed",
                    run_id=run_id,
                    step=step,
                    payload={
                        "results": [asdict(r) for r in results],
                        "raw_llm_response": raw,
                    },
                )
            ]

        case AssistantFinalized(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            raw_llm_response=raw,
            timestamp=timestamp,
        ):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="assistant_finalized",
                    run_id=run_id,
                    step=step,
                    payload={
                        "content": content,
                        "stop_reason": stop_reason,
                        "usage": asdict(usage) if usage else None,
                        "raw_llm_response": raw,
                    },
                )
            ]

        case TurnFinished(done=done, reason=reason, timestamp=timestamp):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="turn_finished",
                    run_id=run_id,
                    step=step,
                    payload={"done": done, "reason": reason},
                )
            ]

        case ControllerRequested(timestamp=timestamp):
            return [
                TraceRecord(
                    timestamp=timestamp,
                    kind="controller_requested",
                    run_id=run_id,
                    step=step,
                    payload=event.get_as_event()["event"],
                )
            ]

        case _:
            return []


async def _run(state: HarnessState, ctx: HarnessContext) -> HarnessState:
    state.step += 1

    async for event in ctx.run_controller(state=state):
        event_entry = event.get_as_event()
        state.event_log.append(event_entry)

        if ctx.trace_sink:
            for trace in _trace_records_from_event(state.state_id, state.step, event):
                await ctx.trace_sink(trace)

        await _emit(ctx, state, "turn_event", event_entry)

        msgs, done = _messages_from_event(event)
        if msgs:
            state.messages.extend(msgs)
        if done:
            state.done = True

    return state


def build_harness_runner(
    step_middlewares: Middlewares, session_middlewares: Middlewares
):
    step_runner = build_chain(step_middlewares, _run)

    async def _session_loop(state: HarnessState, ctx: HarnessContext) -> HarnessState:
        while not state.done:
            state = await step_runner(state, ctx)

        return state

    return build_chain(session_middlewares, _session_loop)


async def run_harness(
    system: str,
    prompt: str | list[Message],
    state: HarnessState,
    ctx: HarnessContext,
    session_runner: Next,
    emit_metadata: Optional[dict] = None,
) -> tuple[Optional[HarnessState], Optional[Exception]]:
    try:
        state.system = system

        await _emit(
            ctx,
            state,
            "run_started",
            {"run_id": state.state_id, **(emit_metadata or {})},
        )

        if isinstance(prompt, str):
            state.prompt = prompt
            state.messages.append(Message(role="user", content=prompt))
        elif isinstance(prompt, list):
            state.messages.extend(prompt)
        else:
            raise ValueError("prompt must be str or list[Message]")

        state = await session_runner(state, ctx)

        await _emit(
            ctx,
            state,
            "run_ended",
            {
                "run_id": state.state_id,
                "reason": "done" if state.done else "stopped",
                **(emit_metadata or {}),
            },
        )
        return state, None
    except Exception as e:
        if state is not None:
            await _emit(
                ctx,
                state,
                "run_failed",
                {"run_id": state.state_id, "error": str(e), **(emit_metadata or {})},
            )
        return state, e
    finally:
        if state is not None:
            await ctx.state_manager.save(state)


def spawn_harness_runner(
    ctx: HarnessContext, middlewares: Middlewares, session_middlewares: Middlewares
):
    session_runner = build_harness_runner(middlewares, session_middlewares)

    async def _runner(
        system: str,
        prompt: str | list[Message],
        state_id: str,
    ):
        state = await ctx.state_manager.load(state_id)

        return await run_harness(
            system=system,
            prompt=prompt,
            state=state,
            ctx=ctx,
            session_runner=session_runner,
            emit_metadata={"branch": False},
        )

    return _runner


def spawn_branch_harness_runner(
    label: str,
    parent_state: HarnessState,
    ctx: HarnessContext,
    middlewares: Middlewares,
    session_middlewares: Middlewares,
    private_of: Optional[str],
):
    branch_id = f"{parent_state.state_id}::{label or str(uuid4())}"
    branch_info = BranchInfo(
        branch_id=branch_id,
        parent_run_id=parent_state.state_id,
        parent_step=parent_state.step,
        spawned_at=time.time(),
        label=label,
    )
    parent_state.branches.append(branch_info)

    branch_ctx = HarnessContext(
        state_manager=ctx.state_manager,
        controller=ctx.controller,
        runtime=ctx.runtime,
        llm=ctx.llm,
        emitter=ctx.emitter,
        trace_sink=ctx.trace_sink,
        private_of=private_of,
    )

    session_runner = build_harness_runner(middlewares, session_middlewares)

    async def _runner(system: str, prompt: str | list[Message]):
        branch_state = await branch_ctx.state_manager.load(branch_id)
        branch_state.branch = branch_info
        return await run_harness(
            system=system,
            prompt=prompt,
            state=branch_state,
            ctx=branch_ctx,
            session_runner=session_runner,
            emit_metadata={
                "branch": True,
                "label": label,
                "parent_run_id": parent_state.state_id,
                "parent_step": parent_state.step,
            },
        )

    return _runner


class Harness:
    def __init__(
        self,
        ctx: HarnessContext,
        step_middlewares: Middlewares,
        session_middlewares: Middlewares,
    ):
        self.ctx = ctx
        self.harness_runner = spawn_harness_runner(
            ctx, step_middlewares, session_middlewares
        )

    async def __aenter__(self):
        await self.ctx.runtime.connect()
        self.ctx._connected = True
        return self

    async def __aexit__(self, *args, **kwargs):
        self.ctx._connected = False
        await self.ctx.runtime.disconnect()

    async def run(
        self, system: str, prompt: str | list[Message], state_id: Optional[str] = None
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        self.ctx._require_connected()
        _state_id = state_id or str(uuid4())
        return await self.harness_runner(system, prompt, _state_id)

    @staticmethod
    def create_branch_runner(
        ctx: HarnessContext,
        label: str,
        parent_state: HarnessState,
        step_middlewares: Optional[Middlewares] = None,
        session_middlewares: Optional[Middlewares] = None,
        private_of: Optional[str] = None,
    ):
        ctx._require_connected()
        return spawn_branch_harness_runner(
            label,
            parent_state,
            ctx,
            step_middlewares or [],
            session_middlewares or [],
            private_of,
        )
