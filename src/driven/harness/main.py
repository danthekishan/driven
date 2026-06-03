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
    AgentEvent,
    ActionEventPayload,
    EventType,
    GenericEventPayload,
    Message,
    ObservationEventPayload,
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
            step_event = AgentEvent(
                GenericEventPayload(type=EventType.STEP_STARTED, data={})
            )
            # Emit externally (if provided) then reduce
            if self.emitter:
                await self.emitter.emit(step_event)
            state = await self.reducer.apply(state, step_event)

            # Fetch tools for this step (dynamic)
            tools = self.runtime.get_tools()

            # request
            action, agent_messages = await self.controller.decide(
                messages=state.messages,
                tools=tools,
                llm=self.llm,
                sink=self.emitter,
            )

            action_event = AgentEvent(
                payload=ActionEventPayload(action=action, agent_messages=agent_messages)
            )
            if self.emitter:
                await self.emitter.emit(action_event)
            state = await self.reducer.apply(state, action_event)

            # execution
            observation, tool_messages = await self.runtime.execute(
                action=action,
                state=state,
                llm=self.llm,
                sink=self.emitter,
            )

            obs_event = AgentEvent(
                ObservationEventPayload(
                    observation=observation, tool_messages=tool_messages
                )
            )
            if self.emitter:
                await self.emitter.emit(obs_event)
            state = await self.reducer.apply(state, obs_event)

            await self.state_manager.save(state)

            if state.done:
                return state
