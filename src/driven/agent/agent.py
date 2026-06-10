from typing import Optional

from driven.core.protocols import Controller, Emitter, Llm, StateManager
from driven.core.harness import Harness, HarnessState, Middlewares
from driven.core.tool_runtime import Extension
from driven.core.schemas import Message


class Agent:
    def __init__(
        self,
        llm: Llm,
        extensions: list[Extension],
        controller: Controller,
        state_manager: StateManager,
        middlewares: Optional[Middlewares] = None,
        session_middlewares: Optional[Middlewares] = None,
        emitter: Optional[Emitter] = None,
    ):
        self.harness = Harness(
            llm=llm,
            extensions=extensions,
            controller=controller,
            state_manager=state_manager,
            emitter=emitter,
            step_middlewares=middlewares or [],
            session_middlewares=session_middlewares or [],
        )

    async def __aenter__(self):
        await self.harness.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.harness.disconnect(exc_type, exc, tb)

    async def run(
        self,
        prompt: str | list[Message],
        system: str = "",
        state_id: Optional[str] = None,
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        return await self.harness.run(system=system, prompt=prompt, state_id=state_id)
