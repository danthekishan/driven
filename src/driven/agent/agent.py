from typing import Optional

from driven.core.protocols import Controller, Emitter, Llm, StateManager
from driven.core.harness import Harness, HarnessState, Middlewares
from driven.core.tool_runtime import Extension
from driven.core.schemas import Message, RunOpts
from driven.agent.controller import ToolCallingController
from driven.agent.state_manager import InMemoryStateManager
from driven.core.harness_middlewares import compaction_step_middleware


class Agent:
    def __init__(
        self,
        llms: list[Llm],
        extensions: list[Extension],
        controller: Optional[Controller] = None,
        state_manager: Optional[StateManager] = None,
        middlewares: Optional[Middlewares] = None,
        session_middlewares: Optional[Middlewares] = None,
        branch_step_middlewares: Optional[Middlewares] = None,
        emitter: Optional[Emitter] = None,
    ):
        self._llms: dict[str, Llm] = {}
        default = ""
        for llm in llms:
            name = llm.name
            if name in self._llms:
                raise RuntimeError(f"Duplicate llm name: '{name}'")
            self._llms[name] = llm
            if not default:
                default = name

        if not default:
            raise RuntimeError("At least one Llm is required")

        self._default_llm = default

        self.harness = Harness(
            llms=self._llms,
            default_llm=self._default_llm,
            extensions=extensions,
            controller=controller or ToolCallingController(),
            state_manager=state_manager or InMemoryStateManager(),
            step_middlewares=middlewares or [],
            session_middlewares=session_middlewares or [],
            branch_step_middlewares=branch_step_middlewares or [compaction_step_middleware()],
            emitter=emitter,
        )

    @property
    def default_llm(self) -> str:
        return self._default_llm

    @property
    def llms(self) -> dict[str, Llm]:
        return dict(self._llms)

    @property
    def state_manager(self) -> StateManager:
        return self.harness.ctx.state_manager

    async def start(self):
        await self.harness.connect()

    async def stop(self):
        await self.harness.disconnect()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.harness.disconnect(exc_type, exc, tb)

    async def run(
        self,
        prompt: str | list[Message],
        system: str = "",
        state_id: Optional[str] = None,
        run_options: Optional[RunOpts] = None,
    ) -> tuple[Optional[HarnessState], Optional[Exception]]:
        return await self.harness.run(
            system=system,
            prompt=prompt,
            state_id=state_id,
            run_options=run_options,
        )

    def cancel(self):
        self.harness.cancel()
