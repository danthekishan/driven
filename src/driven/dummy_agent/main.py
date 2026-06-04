import asyncio

from typing import Optional
from driven.core.tool_runtime import ExtensionRegistry
from driven.dummy_agent.llm import LlmController, ScriptedLlm
from driven.core.harness import (
    Harness,
    HarnessContext,
    HarnessState,
    StateManager,
    Emitter,
)
from driven.core.tool_runtime import Extension, tool
from driven.core.schemas import (
    LlmOutput,
    Message,
    ToolCall,
)


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


class SlackExtension(Extension):
    name = "slack"
    description = "Slack communication tools."

    async def start(self):
        self.client = object()
        print("[SlackExtension] started")

    async def stop(self):
        print("[SlackExtension] stopped")

    @tool(description="Send slack message")
    async def send(self, channel: str, message: str, emitter: Optional[Emitter]):
        if emitter:
            await emitter.emit("slack.send emit", {"hello": "hello from slack.sned"})
        return {
            "channel": channel,
            "message": message,
            "status": "sent",
        }


async def search_tool(query: str):
    return f"search result for: {query}"


async def main():
    llm = ScriptedLlm(
        responses=[
            LlmOutput(
                tool_call=ToolCall(
                    name="slack.send",
                    arguments={"channel": "#general", "message": "hello"},
                )
            ),
            LlmOutput(content="Python is great"),
            LlmOutput(),
        ]
    )

    runtime = ExtensionRegistry(exts=[SlackExtension()])

    ctx = HarnessContext(
        state_manager=InMemoryStateManager(),
        controller=LlmController(),
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
