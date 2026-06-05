import asyncio
from dataclasses import asdict
import json
import os

from dotenv import load_dotenv

from driven.agent.controller import ToolCallingController
from driven.core.harness import (
    Emitter,
    Harness,
    HarnessContext,
    HarnessState,
    StateManager,
)
from driven.core.harness_middlewares import (
    compaction_step_middleware,
    lifecycle_session_middleware,
    lifecycle_step_middleware,
    max_steps_middleware,
)
from driven.core.tool_runtime import ExtensionRegistry
from driven.extensions.coder import CoderExtension
from driven.llm_providers.openai import OpenAILlm


# =========================
# STATE MANAGER
# =========================


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

    llm = OpenAILlm(model="gpt-4.1-mini", api_key=api_key)
    runtime = ExtensionRegistry(exts=[CoderExtension(workspace="./workspace")])

    # =========================
    # CONTROLLER
    # =========================

    controller = ToolCallingController()

    # =========================
    # CONTEXT
    # =========================

    ctx = HarnessContext(
        state_manager=(InMemoryStateManager()),
        controller=controller,
        runtime=runtime,
        llm=llm,
        emitter=PrintEmitter(),
    )

    # =========================
    # HARNESS
    # =========================

    async with Harness(
        ctx=ctx,
        step_middlewares=[
            max_steps_middleware(20),
            lifecycle_step_middleware(),
            compaction_step_middleware(),
        ],
        session_middlewares=[lifecycle_session_middleware(run_id="run-1")],
    ) as harness:
        state, error = await harness.run(
            system="",
            run_id="run-1",
            prompt=("Create a hello.py file that prints hello world, then run it."),
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

    print("====================")
    if state:
        with open("xxx.json", "w") as f:
            f.write(json.dumps(asdict(state), indent=2))
    print("====================")


if __name__ == "__main__":
    asyncio.run(main())
