from uuid import uuid4
from functools import wraps
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Literal, Optional, Protocol, Union

from driven.harness.protocols import (
    Controller,
    Emitter,
    Runtime,
    Llm,
    StateManager,
    Reducer,
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
    reducer: Reducer
    emitter: Optional[Emitter] = None


type Next = Callable[[AgentState, Context], Awaitable[AgentState]]


class Middleware(Protocol):
    async def __call__(
        self, state: AgentState, ctx: Context, next: Next
    ) -> AgentState: ...


async def _run(state: AgentState, ctx: Context) -> AgentState:
    state.step += 1

    actions = await ctx.controller.decide(
        messages=state.messages,
        get_tools=ctx.runtime.get_tools,
        llm=ctx.llm,
        sink=ctx.emitter,
    )

    observations: list[ObservationEvent] = []

    for action in actions:
        if ctx.emitter:
            await ctx.emitter.emit(action)

        if isinstance(action, RespondAction):
            observations.append(AssistantResp(content=action.content))
        elif isinstance(action, CallToolAction):
            obs = await ctx.runtime.execute(
                action=action, state=state, llm=ctx.llm, sink=ctx.emitter
            )
            observations.extend(obs)
        else:
            raise RuntimeError(f"Unknown action type: {type(action)}")

    if ctx.emitter and observations:
        await ctx.emitter.emit(observations)

    # for event in observations:
    #     match event:
    #         case AssistantResp():

    state = await ctx.reducer.apply(state, observations)
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

    async def run(self):
        await self.startup()

        run_id = uuid4()
        state = await self.ctx.state_manager.load(str(run_id))

        try:
            state = await self.session_runner(state, self.ctx)
            return state
        finally:
            await self.ctx.state_manager.save(state)
            await self.shutdown()
