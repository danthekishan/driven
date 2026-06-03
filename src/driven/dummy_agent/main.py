import asyncio

from driven.dummy_agent.llm import LlmController, ScriptedLlm
from driven.dummy_agent.protocols import (
    PrintEmitter,
    SimpleReducer,
    DummyRuntime,
    InMemoryStateManager,
)
from driven.harness.main import Harness
from driven.harness.types import (
    LlmOutput,
    Message,
    ToolCall,
)


async def search_tool(
    query: str,
):
    return f"search result for: {query}"


async def main():

    llm = ScriptedLlm(
        responses=[
            LlmOutput(
                tool_call=ToolCall(
                    name="search",
                    arguments={
                        "query": "python",
                    },
                )
            ),
            LlmOutput(
                content="Python is great",
            ),
            LlmOutput(),
        ]
    )

    runtime = DummyRuntime()

    runtime.register_tool(
        name="search",
        fn=search_tool,
        description="Search something",
    )

    harness = Harness(
        state_manager=InMemoryStateManager(),
        llm=llm,
        reducer=SimpleReducer(),
        controller=LlmController(),
        runtime=runtime,
        emitter=PrintEmitter(),
    )

    state = await harness.run(
        run_id="run-1",
        input=[
            Message(
                role="user",
                content="Tell me about python",
            )
        ],
    )

    print()
    print("FINAL STATE")
    print(state)


if __name__ == "__main__":
    asyncio.run(main())
