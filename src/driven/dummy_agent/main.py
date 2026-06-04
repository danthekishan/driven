import asyncio

from typing import Optional
from driven.agent.controller import ToolCallingController
from driven.core.tool_runtime import ExtensionRegistry
from driven.core.harness import (
    Harness,
    HarnessContext,
    HarnessState,
    StateManager,
    Emitter,
)
from driven.core.schemas import Message
from driven.extensions.coder import CoderExtension
from driven.llm_providers.openai import OpenAILlm


class InMemoryStateManager(StateManager):
    def __init__(self):
        self.store: dict[str, HarnessState] = {}

    async def load(self, run_id: str) -> HarnessState:
        if run_id in self.store:
            return self.store[run_id]

        state = HarnessState(state_id=run_id, prompt="", messages=[])
        self.store[run_id] = state
        return state

    async def save(self, state: HarnessState) -> HarnessState:
        self.store[state.state_id] = state
        return state


class PrintEmitter(Emitter):
    async def emit(self, event_type, events) -> None:
        # Accept single or list
        if isinstance(events, (list, tuple)):
            for e in events:
                print(event_type, e)
        else:
            print(event_type, events)


async def main():
    llm = OpenAILlm()

    runtime = ExtensionRegistry(exts=[CoderExtension()])

    ctx = HarnessContext(
        state_manager=InMemoryStateManager(),
        controller=ToolCallingController(),
        runtime=runtime,
        llm=llm,
        emitter=PrintEmitter(),
    )

    # Middlewares: lifecycle at session level, compaction at step level
    from driven.core.harness_middlewares import (
        lifecycle_session_middleware,
        lifecycle_step_middleware,
        compaction_step_middleware,
    )

    async with Harness(
        ctx=ctx,
        step_middlewares=[
            lifecycle_step_middleware(),
            compaction_step_middleware(max_messages=6, keep_last=4),
        ],
        session_middlewares=[lifecycle_session_middleware(run_id="run-1")],
    ) as harness:
        # Inject initial prompt and input; harness will set prompt/messages on new state
        user_prompt = "Tell me about python"
        user_input = [Message(role="user", content=user_prompt)]

        state = await harness.run(run_id="run-1", prompt=user_prompt, input=user_input)

        print()
        print("FINAL STATE")
        print(state)


if __name__ == "__main__":
    asyncio.run(main())
