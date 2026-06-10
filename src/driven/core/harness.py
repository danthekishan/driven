from functools import wraps
from dataclasses import asdict
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Protocol,
)
from uuid import uuid4

from driven.core.schemas import (
    AssistantFinalized,
    BranchInfo,
    ControllerRequested,
    LlmOutputReceived,
    LlmRequestPrepared,
    Message,
    ToolsCompleted,
    ToolsRequested,
    TraceRecord,
    TurnEvent,
    TurnFinished,
)
from driven.core.protocols import (
    HarnessState,
    Emitter,
    Llm,
    Controller,
    StateManager,
    HarnessContext,
)
from driven.core.tool_runtime import Extension, ToolRuntime


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
        llm: Llm,
        extensions: list[Extension],
        controller: Controller,
        state_manager: StateManager,
        step_middlewares: Middlewares,
        session_middlewares: Middlewares,
        emitter: Optional[Emitter] = None,
        max_start_attempts: int = 3,
        retry_delay: float = 2.0,
    ):

        runtime = ToolRuntime(
            exts=extensions,
            max_start_attempts=max_start_attempts,
            retry_delay=retry_delay,
        )

        self.ctx = HarnessContext(
            state_manager=state_manager,
            controller=controller,
            runtime=runtime,
            llm=llm,
            emitter=emitter,
        )

        self.harness_runner = spawn_harness_runner(
            self.ctx, step_middlewares, session_middlewares
        )

    async def connect(self):
        async def _create_branch(
            label: str,
            parent_state: HarnessState,
            middlewares: Middlewares,
            session_middlewares: Middlewares,
            private_of: Optional[str],
        ):
            self.ctx._require_connected()
            return spawn_branch_harness_runner(
                label,
                parent_state,
                self.ctx,
                middlewares,
                session_middlewares,
                private_of,
            )

        await self.ctx.runtime.connect(_create_branch)
        self.ctx._connected = True
        return self

    async def disconnect(self, *args, **kwargs):
        self.ctx._connected = False
        await self.ctx.runtime.disconnect()

    async def run(
        self, system: str, prompt: str | list[Message], state_id: Optional[str] = None
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        self.ctx._require_connected()
        _state_id = state_id or str(uuid4())
        return await self.harness_runner(system, prompt, _state_id)
