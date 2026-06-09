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


@runtime_checkable
class Emitter(Protocol):
    async def emit(self, event: dict[str, Any]) -> None: ...


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


SpawnBranch = Callable[
    ...,
    Awaitable[tuple[Optional["HarnessState"], Optional[Exception]]],
]


@runtime_checkable
class Runtime(Protocol):
    async def __aenter__(self) -> Any: ...
    async def __aexit__(self, exc_type, exc, tb): ...
    def get_tools(self, *args, **kwargs) -> list[LlmToolFunction]: ...
    async def call_tools(
        self,
        tools: list[ToolCall],
        state: HarnessState,
        llm: Llm,
        emitter: Optional[Emitter] = None,
        timeout: Optional[float] = None,
        spawn_branch: Optional[SpawnBranch] = None,
    ) -> list[ToolResult]: ...


CallTools = Callable[[list[ToolCall], Optional[float]], Awaitable[list[ToolResult]]]


@runtime_checkable
class Controller(Protocol):
    def stream_turn(
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
    async def record(self, trace: TraceRecord) -> None: ...


@dataclass
class HarnessContext:
    state_manager: StateManager
    controller: Controller
    runtime: Runtime
    llm: Llm
    emitter: Optional[Emitter] = None
    trace_sink: Optional[TraceSink] = None
    private_of: Optional[str] = None


async def _emit(
    ctx: HarnessContext,
    state: HarnessState,
    name: str,
    payload: dict[str, Any],
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
    await ctx.emitter.emit(event)


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
    run_id: str,
    step: int,
    event: TurnEvent,
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

    async def spawn_branch(
        parent_state: HarnessState,
        prompt: str | list[Message],
        label: str = "",
        system: str = "",
        middlewares: Middlewares = [],
        session_middlewares: Middlewares = [],
        private_of: Optional[str] = None,
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
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

        branch_state: Optional[HarnessState] = None
        try:
            branch_state = await branch_ctx.state_manager.load(branch_id)
            branch_state.branch = branch_info
            branch_state.system = system

            if isinstance(prompt, str):
                branch_state.prompt = prompt
                branch_state.messages.append(Message(role="user", content=prompt))
            elif isinstance(prompt, list):
                branch_state.messages.extend(prompt)

            await _emit(
                branch_ctx, branch_state, "branch_started",
                {"branch_id": branch_id, "label": label, "parent_run_id": parent_state.state_id, "parent_step": parent_state.step},
            )

            branch_step_runner = build_chain(middlewares, _run) if middlewares else _run

            async def _branch_session(state: HarnessState, ctx: HarnessContext) -> HarnessState:
                return await _run_loop(state, ctx, branch_step_runner)

            branch_session_runner = build_chain(session_middlewares, _branch_session) if session_middlewares else _branch_session

            branch_state = await branch_session_runner(branch_state, branch_ctx)

            await _emit(
                branch_ctx, branch_state, "branch_ended",
                {"branch_id": branch_id, "reason": "done" if branch_state.done else "stopped"},
            )

            return branch_state, None
        except Exception as e:
            if branch_state is not None:
                await _emit(
                    branch_ctx, branch_state, "branch_failed",
                    {"branch_id": branch_id, "error": str(e)},
                )
            return branch_state, e
        finally:
            if branch_state is not None:
                await ctx.state_manager.save(branch_state)

    async def call_tools(
        tool_calls: list[ToolCall],
        timeout: Optional[float] = None,
    ) -> list[ToolResult]:
        return await ctx.runtime.call_tools(
            tools=tool_calls,
            state=state,
            llm=ctx.llm,
            emitter=ctx.emitter,
            timeout=timeout,
            spawn_branch=spawn_branch,
        )

    def get_tools() -> list[LlmToolFunction]:
        return ctx.runtime.get_tools(private_of=ctx.private_of)

    async for event in ctx.controller.stream_turn(
        messages=state.messages,
        system=state.system,
        get_tools=get_tools,
        call_tools=call_tools,
        llm=ctx.llm,
    ):
        event_entry = event.get_as_event()
        state.event_log.append(event_entry)

        if ctx.trace_sink:
            for trace in _trace_records_from_event(state.state_id, state.step, event):
                await ctx.trace_sink.record(trace)

        await _emit(ctx, state, "turn_event", event_entry)

        msgs, done = _messages_from_event(event)
        if msgs:
            state.messages.extend(msgs)
        if done:
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


async def _run_loop(
    state: HarnessState,
    ctx: HarnessContext,
    step_runner: Next,
) -> HarnessState:
    while not state.done:
        state = await step_runner(state, ctx)

    return state


class Harness:
    def __init__(
        self,
        ctx: HarnessContext,
        step_middlewares: Middlewares,
        session_middlewares: Middlewares,
    ):
        self.ctx = ctx
        self.step_runner = build_chain(step_middlewares, _run)

        async def _session_loop(state: HarnessState, ctx: HarnessContext) -> HarnessState:
            return await _run_loop(state, ctx, self.step_runner)

        self.session_runner = build_chain(session_middlewares, _session_loop)

    async def __aenter__(self):
        await self.ctx.runtime.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.ctx.runtime.__aexit__(exc_type, exc, tb)

    async def run(
        self,
        system: str,
        prompt: str | list[Message],
        run_id: Optional[str] = None,
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        rid = str(run_id or uuid4())
        state: Optional[HarnessState] = None
        try:
            state = await self.ctx.state_manager.load(rid)
            state.system = system

            await _emit(self.ctx, state, "run_started", {"run_id": state.state_id})

            if isinstance(prompt, str):
                state.prompt = prompt
                state.messages.append(Message(role="user", content=prompt))
            elif isinstance(prompt, list):
                state.messages.extend(prompt)
            else:
                raise ValueError("prompt must be str or list[Message]")

            state = await self.session_runner(state, self.ctx)

            await _emit(
                self.ctx,
                state,
                "run_ended",
                {
                    "run_id": state.state_id,
                    "reason": "done" if state.done else "stopped",
                },
            )
            return state, None
        except Exception as e:
            if state is not None:
                await _emit(
                    self.ctx,
                    state,
                    "run_failed",
                    {"run_id": state.state_id, "error": str(e)},
                )
            return state, e
        finally:
            if state is not None:
                await self.ctx.state_manager.save(state)
