from driven.core.harness import HarnessContext, HarnessState, Middleware, Next
from driven.core.schemas import Message


def compaction_step_middleware(
    max_messages: int = 20, keep_last: int = 12
) -> Middleware:
    async def mw(state: HarnessState, ctx: HarnessContext, next: Next) -> HarnessState:
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
            total = int(new_state.internal.get("compacted_total", 0)) + drop
            new_state.internal["compacted_total"] = total
        return new_state

    return mw


def lifecycle_session_middleware(run_id: str | None = None) -> Middleware:
    async def mw(state: HarnessState, ctx: HarnessContext, next: Next) -> HarnessState:
        return await next(state, ctx)

    return mw


def lifecycle_step_middleware() -> Middleware:
    async def mw(state: HarnessState, ctx: HarnessContext, next: Next) -> HarnessState:
        return await next(state, ctx)

    return mw


def max_steps_middleware(
    max_steps: int = 20,
) -> Middleware:
    async def mw(
        state: HarnessState,
        ctx: HarnessContext,
        next: Next,
    ) -> HarnessState:
        if state.step >= max_steps:
            state.done = True
            state.messages.append(
                Message(
                    role="assistant",
                    content=(f"Stopped after {max_steps} steps."),
                )
            )
            return state
        return await next(state, ctx)

    return mw
