import asyncio
from dataclasses import asdict
import json
import os

from dotenv import load_dotenv

from driven.agent import Agent, InMemoryStateManager, ToolCallingController
from driven.core.harness import Emitter
from driven.core.harness_middlewares import (
    compaction_step_middleware,
    lifecycle_session_middleware,
    lifecycle_step_middleware,
    max_steps_middleware,
)
from driven.core.tool_runtime import ExtensionRegistry
from coder import CoderExtension
from driven.llm_providers.openai import OpenAILlm


class PrintEmitter(Emitter):
    async def emit(self, event: dict) -> None:
        source = event.get("source", "unknown")
        name = event.get("name", "event")
        payload = event.get("payload", {})
        print(f"[{source}.{name}] {payload}")


async def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    async with Agent(
        llm=OpenAILlm(model="gpt-4.1-mini", api_key=api_key),
        runtime=ExtensionRegistry(exts=[CoderExtension(workspace="./workspace")]),
        controller=ToolCallingController(),
        state_manager=InMemoryStateManager(),
        middlewares=[
            max_steps_middleware(20),
            lifecycle_step_middleware(),
            compaction_step_middleware(),
        ],
        session_middlewares=[lifecycle_session_middleware(run_id="run-1")],
        emitter=PrintEmitter(),
    ) as agent:
        state, error = await agent.run(
            prompt="Create a hello.py file that prints hello world, then run it.",
            run_id="run-1",
        )

        if error:
            print(error)
            return

        if state:
            print()
            print("=" * 80)
            print("FINAL STATE")
            print("=" * 80)
            print()

            print("DONE:", state.done)
            print("STEP:", state.step)

            print()
            print("MESSAGES")
            print("-" * 80)

            for msg in state.messages:
                print(f"[{msg.role}] {msg.content}")
                print()

    if state:
        with open("xxx.json", "w") as f:
            f.write(json.dumps(asdict(state), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
