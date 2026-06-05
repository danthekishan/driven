from typing import Optional
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
            # Track simple counter
            total = int(new_state.metadata.get("compacted_total", 0)) + drop
            new_state.metadata["compacted_total"] = total
        return new_state

    return mw


def lifecycle_session_middleware(run_id: Optional[str] = None) -> Middleware:
    async def mw(state: HarnessState, ctx: HarnessContext, next: Next) -> HarnessState:
        # Emit run_started
        if ctx.emitter:
            await ctx.emitter.emit("run_started", {"run_id": run_id or state.state_id})
        # Run loop
        new_state = await next(state, ctx)
        # Emit run_ended
        if ctx.emitter:
            await ctx.emitter.emit(
                "run_ended",
                {
                    "run_id": run_id or state.state_id,
                    "reason": "done" if new_state.done else "stopped",
                },
            )
        return new_state

    return mw


def lifecycle_step_middleware() -> Middleware:
    async def mw(state: HarnessState, ctx: HarnessContext, next: Next) -> HarnessState:
        prev_step = state.step
        new_state = await next(state, ctx)
        if ctx.emitter and new_state.step != prev_step:
            await ctx.emitter.emit("step_advanced", {"step": new_state.step})
        return new_state

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
