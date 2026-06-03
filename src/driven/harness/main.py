from typing import Optional

from driven.harness.protocols import (
    Controller,
    Emitter,
    Runtime,
    Llm,
    StateManager,
    Reducer,
)
from driven.harness.types import (
    Message,
    RespondAction,
    CallToolAction,
    AssistantSaid,
    ObservationEvent,
)


class Harness:
    def __init__(
        self,
        state_manager: StateManager,
        llm: Llm,
        reducer: Reducer,
        controller: Controller,
        runtime: Runtime,
        emitter: Optional[Emitter] = None,
    ):
        self.state_manager = state_manager
        self.reducer = reducer
        self.llm = llm
        self.emitter = emitter
        self.controller = controller
        self.runtime = runtime

    async def run(self, run_id: str, input: list[Message] = []):
        state = await self.state_manager.load(run_id, input)

        while True:
            # Step increment inside harness per current design
            state.step += 1

            # Controller decides a sequence of actions, with dynamic tool access
            actions = await self.controller.decide(
                messages=state.messages,
                get_tools=self.runtime.get_tools,
                llm=self.llm,
                sink=self.emitter,
            )

            # Execute actions and reduce observations
            for action in actions:
                if self.emitter:
                    await self.emitter.emit(action)

                observations: list[ObservationEvent]

                if isinstance(action, RespondAction):
                    observations = [AssistantSaid(content=action.content)]
                elif isinstance(action, CallToolAction):
                    observations = await self.runtime.execute(
                        action=action,
                        state=state,
                        llm=self.llm,
                        sink=self.emitter,
                    )
                else:
                    raise RuntimeError(f"Unknown action type: {type(action)}")

                if self.emitter:
                    await self.emitter.emit(observations)
                state = await self.reducer.apply(state, observations)

            await self.state_manager.save(state)

            if state.done:
                return state
