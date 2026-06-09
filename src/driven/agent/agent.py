from typing import Optional

from driven.core.harness import (
    Controller,
    Emitter,
    Harness,
    HarnessContext,
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
        self.llm = llm
        self.runtime = runtime
        self.controller = controller
        self.state_manager = state_manager
        self.middlewares = middlewares
        self.session_middlewares = session_middlewares
        self.emitter = emitter

    async def __aenter__(self):
        await self.runtime.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.runtime.__aexit__(exc_type, exc, tb)

    async def run(
        self,
        prompt: str | list[Message],
        system: str = "",
        run_id: Optional[str] = None,
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        ctx = HarnessContext(
            state_manager=self.state_manager,
            controller=self.controller,
            runtime=self.runtime,
            llm=self.llm,
            emitter=self.emitter,
        )

        harness = Harness(
            ctx=ctx,
            step_middlewares=self.middlewares,
            session_middlewares=self.session_middlewares,
        )

        return await harness.run(system=system, prompt=prompt, run_id=run_id)
