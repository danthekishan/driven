from uuid import uuid4
from functools import wraps
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

from driven.harness.protocols import (
    Controller,
    Emitter,
    Runtime,
    Llm,
    StateManager,
)
from driven.harness.types import (
    AgentState,
    AssistantResp,
    Message,
    RespondAction,
    CallToolAction,
    ObservationEvent,
)


# class Harness:
#     def __init__(
#         self,
#         state_manager: StateManager,
#         llm: Llm,
#         reducer: Reducer,
#         controller: Controller,
#         runtime: Runtime,
#         emitter: Optional[Emitter] = None,
#     ):
#         self.state_manager = state_manager
#         self.reducer = reducer
#         self.llm = llm
#         self.emitter = emitter
#         self.controller = controller
#         self.runtime = runtime
#
#     async def run(self, run_id: str, input: list[Message] = []):
#         state = await self.state_manager.load(run_id, input)
#
#         # Emit run start
#         if self.emitter:
#             await self.emitter.emit(RunStarted(run_id=run_id))
#
#         while True:
#             # Step increment inside harness per current design
#             state.step += 1
#             # Emit step advanced
#             if self.emitter:
#                 await self.emitter.emit(StepAdvanced(step=state.step))
#
#             # Controller decides a sequence of actions, with dynamic tool access
#             actions = await self.controller.decide(
#                 messages=state.messages,
#                 get_tools=self.runtime.get_tools,
#                 llm=self.llm,
#                 sink=self.emitter,
#             )
#
#             # Execute actions and reduce observations
#             for action in actions:
#                 if self.emitter:
#                     await self.emitter.emit(action)
#
#                 observations: list[ObservationEvent]
#
#                 if isinstance(action, RespondAction):
#                     observations = [AssistantSaid(content=action.content)]
#                 elif isinstance(action, CallToolAction):
#                     observations = await self.runtime.execute(
#                         action=action,
#                         state=state,
#                         llm=self.llm,
#                         sink=self.emitter,
#                     )
#                 else:
#                     raise RuntimeError(f"Unknown action type: {type(action)}")
#
#                 if self.emitter:
#                     await self.emitter.emit(observations)
#                 state = await self.reducer.apply(state, observations)
#
#             await self.state_manager.save(state)
#
#             if state.done:
#                 if self.emitter:
#                     await self.emitter.emit(RunEnded(run_id=run_id, reason="done"))
#                 return state


@dataclass
class Context:
    state_manager: StateManager
    controller: Controller
    runtime: Runtime
    llm: Llm
    emitter: Optional[Emitter] = None


type Next = Callable[[AgentState, Context], Awaitable[AgentState]]


class Middleware(Protocol):
    async def __call__(
        self, state: AgentState, ctx: Context, next: Next
    ) -> AgentState: ...


async def _run(state: AgentState, ctx: Context) -> AgentState:
    # Advance step
    state.step += 1

    # Decide actions with dynamic tools
    actions = await ctx.controller.decide(
        messages=state.messages,
        get_tools=ctx.runtime.get_tools,
        llm=ctx.llm,
        sink=ctx.emitter,
    )

    observations: list[ObservationEvent] = []

    # Emit actions as a batch (if any)
    if ctx.emitter and actions:
        await ctx.emitter.emit(actions)

    # Map actions to observations
    for action in actions:
        if isinstance(action, RespondAction):
            observations.append(AssistantResp(content=action.content))
        elif isinstance(action, CallToolAction):
            obs = await ctx.runtime.execute(
                action=action, state=state, llm=ctx.llm, sink=ctx.emitter
            )
            observations.extend(obs)
        else:
            raise RuntimeError(f"Unknown action type: {type(action)}")

    # Emit observations as a batch (if any)
    if ctx.emitter and observations:
        await ctx.emitter.emit(observations)

    # Capture recents/logs
    state.recent_actions = actions
    state.recent_observations = observations
    state.actions_log.append(actions)
    state.observations_log.append(observations)

    # Update transcript and minimal stop policy
    for event in observations:
        if isinstance(event, AssistantResp):
            state.messages.append(Message(role="assistant", content=event.content))
            if event.content.strip() == "":
                state.done = True
        elif hasattr(event, "tool_name") and hasattr(event, "content"):
            # ToolProduced
            state.messages.append(
                Message(
                    role="tool",
                    name=getattr(event, "tool_name"),
                    content=str(getattr(event, "content")),
                )
            )
        elif hasattr(event, "tool_name") and hasattr(event, "error_type"):
            # ToolFailed
            state.messages.append(
                Message(
                    role="tool",
                    name=getattr(event, "tool_name"),
                    content=f"[{getattr(event, 'error_type')}] {getattr(event, 'message')}",
                )
            )
        else:
            pass

    return state


def wrap_middleware(middleware, next):
    @wraps(next)
    async def wrapped(state, ctx):
        return await middleware(state, ctx, next)

    return wrapped


def build_chain(middlewares, handler):
    chain = handler

    for mw in reversed(middlewares):
        chain = wrap_middleware(mw, chain)

    return chain


# ---- Example middlewares ----

def compaction_step_middleware(max_messages: int = 20, keep_last: int = 12) -> Middleware:
    async def mw(state: AgentState, ctx: Context, next: Next) -> AgentState:
        new_state = await next(state, ctx)
        if new_state.done:
            return new_state
        msgs = new_state.messages
        if len(msgs) > max_messages:
            drop = len(msgs) - keep_last
            compaction_note = Message(
                role="assistant",
                content=f"[compacted {drop} messages]",
            )
            new_state.messages = [compaction_note] + msgs[-keep_last:]
            # Track simple counter
            total = int(new_state.metadata.get("compacted_total", 0)) + drop
            new_state.metadata["compacted_total"] = total
        return new_state

    return mw


def lifecycle_session_middleware(run_id: Optional[str] = None) -> Middleware:
    async def mw(state: AgentState, ctx: Context, next: Next) -> AgentState:
        # Emit run_started
        if ctx.emitter:
            await ctx.emitter.emit({"type": "run_started", "run_id": run_id or state.state_id})
        # Run loop
        new_state = await next(state, ctx)
        # Emit run_ended
        if ctx.emitter:
            await ctx.emitter.emit({"type": "run_ended", "run_id": run_id or state.state_id, "reason": "done" if new_state.done else "stopped"})
        return new_state

    return mw


def lifecycle_step_middleware() -> Middleware:
    async def mw(state: AgentState, ctx: Context, next: Next) -> AgentState:
        prev_step = state.step
        new_state = await next(state, ctx)
        if ctx.emitter and new_state.step != prev_step:
            await ctx.emitter.emit({"type": "step_advanced", "step": new_state.step})
        return new_state

    return mw


class Harness:
    def __init__(self, ctx: Context, step_middlewares, session_middlewares):
        self.ctx = ctx
        self.step_runner = build_chain(step_middlewares, _run)
        self.session_runner = build_chain(session_middlewares, self._run_loop)

    async def startup(self):
        pass

    async def shutdown(self):
        pass

    async def _run_loop(self, state: AgentState, ctx: Context):
        while not state.done:
            state = await self.step_runner(state, ctx)

        return state

    async def run(
        self,
        run_id: Optional[str] = None,
        prompt: str = "",
        input: list[Message] | None = None,
    ):
        await self.startup()

        rid = str(run_id or uuid4())
        state: Optional[AgentState] = None
        try:
            state = await self.ctx.state_manager.load(rid)

            # Inject initial prompt and input if this looks like a new state
            if (
                state.step == 0
                and not state.messages
                and not getattr(state, "prompt", "")
            ):
                state.prompt = prompt
            if input:
                state.messages.extend(input)

            state = await self.session_runner(state, self.ctx)
            return state
        finally:
            if state is not None:
                await self.ctx.state_manager.save(state)
            await self.shutdown()
