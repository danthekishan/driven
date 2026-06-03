import asyncio

from driven.dummy_agent.llm import LlmController, ScriptedLlm
from driven.dummy_agent.protocols import (
    PrintEmitter,
    DummyRuntime,
    InMemoryStateManager,
)
from driven.harness.main import Harness, Context
from driven.harness.types import (
    LlmOutput,
    Message,
    ToolCall,
)


async def search_tool(query: str):
    return f"search result for: {query}"


async def main():
    llm = ScriptedLlm(
        responses=[
            LlmOutput(tool_call=ToolCall(name="search", arguments={"query": "python"})),
            LlmOutput(content="Python is great"),
            LlmOutput(),
        ]
    )

    runtime = DummyRuntime()
    runtime.register_tool(name="search", fn=search_tool, description="Search something")

    ctx = Context(
        state_manager=InMemoryStateManager(),
        controller=LlmController(),
        runtime=runtime,
        llm=llm,
        emitter=PrintEmitter(),
    )

    # Middlewares: lifecycle at session level, compaction at step level
    from driven.harness.main import (
        lifecycle_session_middleware,
        lifecycle_step_middleware,
        compaction_step_middleware,
    )

    harness = Harness(
        ctx=ctx,
        step_middlewares=[
            lifecycle_step_middleware(),
            compaction_step_middleware(max_messages=6, keep_last=4),
        ],
        session_middlewares=[lifecycle_session_middleware(run_id="run-1")],
    )

    # Inject initial prompt and input; harness will set prompt/messages on new state
    user_prompt = "Tell me about python"
    user_input = [Message(role="user", content=user_prompt)]

    state = await harness.run(run_id="run-1", prompt=user_prompt, input=user_input)

    print()
    print("FINAL STATE")
    print(state)


if __name__ == "__main__":
    asyncio.run(main())
