from typing import Optional

from driven.core.harness import (
    Controller,
    Emitter,
    Harness,
    HarnessState,
    Llm,
    Middlewares,
    Runtime,
    StateManager,
)
from driven.core.schemas import Message


class Agent:
    def __init__(
        self,
        llm: Llm,
        runtime: Runtime,
        controller: Controller,
        state_manager: StateManager,
        middlewares: Middlewares = [],
        session_middlewares: Middlewares = [],
        emitter: Optional[Emitter] = None,
    ):
        self.middlewares = middlewares
        self.session_middlewares = session_middlewares

        self.harness = Harness(
            llm=llm,
            runtime=runtime,
            controller=controller,
            state_manager=state_manager,
            emitter=emitter,
            step_middlewares=self.middlewares,
            session_middlewares=self.session_middlewares,
        )

    async def __aenter__(self):
        await self.harness.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.harness.disconnect()

    async def run(
        self,
        prompt: str | list[Message],
        system: str = "",
        state_id: Optional[str] = None,
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        return await self.harness.run(system=system, prompt=prompt, state_id=state_id)
