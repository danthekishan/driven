import argparse
import asyncio
import os

from dotenv import load_dotenv

from driven.agent import Agent, InMemoryStateManager, ToolCallingController
from driven.core.protocols import Emitter
from driven.core.harness_middlewares import (
    compaction_step_middleware,
    max_steps_middleware,
)
from driven.llm_providers.openai import OpenAILlm

from guess import NumberGuessExtension
from coding_agent import CodingAgent


class PrintEmitter(Emitter):
    async def __call__(self, event: dict) -> None:
        source = event.get("source", "unknown")
        name = event.get("name", "event")
        payload = event.get("payload", {})
        print(f"[{source}.{name}] {payload}")


def _print_result(state, label="RESULT"):
    if not state:
        print("No state returned.")
        return

    print()
    print("=" * 80)
    print(f" {label}")
    print("=" * 80)
    print()
    print("DONE:", state.done)
    print("STEP:", state.step)

    if state.branch:
        print(
            "BRANCH:",
            state.branch.branch_id,
            f"(parent: {state.branch.parent_run_id}, step: {state.branch.parent_step})",
        )

    if state.branches:
        print("BRANCHES SPAWNED:", len(state.branches))
        for b in state.branches:
            print(f"  - {b.branch_id} (label: {b.label or '-'})")

    print()
    print("MESSAGES")
    print("-" * 80)
    for msg in state.messages:
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"[{msg.role}] {content}")
        print()


async def run_number_guess(api_key: str):
    async with Agent(
        llm=OpenAILlm(model="gpt-4.1-mini", api_key=api_key),
        extensions=[NumberGuessExtension()],
        controller=ToolCallingController(),
        state_manager=InMemoryStateManager(),
        middlewares=[
            max_steps_middleware(20),
            compaction_step_middleware(),
        ],
        emitter=PrintEmitter(),
    ) as agent:
        state, error = await agent.run(
            prompt="Play the number guessing game. Start a new game and find the number.",
            state_id="run-guess",
        )

        if error:
            print(f"Error: {error}")
            return

        _print_result(state, "NUMBER GUESS")


async def run_coding_agent(api_key: str):
    async with Agent(
        llm=OpenAILlm(model="gpt-4.1-mini", api_key=api_key),
        extensions=[CodingAgent()],
        controller=ToolCallingController(),
        state_manager=InMemoryStateManager(),
        middlewares=[
            max_steps_middleware(20),
            compaction_step_middleware(),
        ],
        emitter=PrintEmitter(),
    ) as agent:
        state, error = await agent.run(
            prompt="Create a fibonacci.py that computes the 10th fibonacci number, then run it.",
            state_id="run-coding-agent",
        )

        if error:
            print(f"Error: {error}")
            return

        _print_result(state, "CODING AGENT (BRANCH)")


EXAMPLES = {
    "number-guess": run_number_guess,
    "coding-agent": run_coding_agent,
}


async def main():
    parser = argparse.ArgumentParser(description="Driven examples")
    parser.add_argument(
        "--name",
        choices=list(EXAMPLES.keys()),
        required=True,
        help="Example to run",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")

    await EXAMPLES[args.name](api_key)


if __name__ == "__main__":
    asyncio.run(main())
