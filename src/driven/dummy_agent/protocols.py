from typing import Any, Optional, Sequence, Union
from driven.harness.protocols import StateManager, Emitter, Llm, Runtime, Reducer
from driven.harness.types import (
    AgentState,
    Message,
    CallToolAction,
    AssistantSaid,
    ObservationEvent,
    ToolProduced,
    ToolFailed,
    LlmToolFunction,
)


class InMemoryStateManager(StateManager):
    def __init__(self):
        self.store: dict[str, AgentState] = {}

    async def load(self, run_id: str, input: list[Message]) -> AgentState:
        if run_id in self.store:
            existing_state = self.store[run_id]
            # Merge new input onto messages without losing previous state
            existing_state.messages.extend(input)
            return existing_state

        state = AgentState(state_id=run_id, messages=input.copy())
        self.store[run_id] = state
        return state

    async def save(self, state: AgentState) -> AgentState:
        self.store[state.state_id] = state
        return state


class PrintEmitter(Emitter):
    async def emit(self, events) -> None:
        # Accept single or list
        if isinstance(events, (list, tuple)):
            for e in events:
                print(e)
        else:
            print(events)


class DummyRuntime(Runtime):
    def __init__(self):
        self.tools: dict[str, Any] = {}

    def register_tool(self, name: str, fn, description: str = ""):
        self.tools[name] = {
            "fn": fn,
            "description": description,
        }

    def get_tools(self, *args, **kwargs) -> list[LlmToolFunction]:
        output = []

        for name, tool in self.tools.items():
            output.append(
                LlmToolFunction(
                    name=name,
                    description=tool["description"],
                    parameters_schema={},
                )
            )

        return output

    async def execute(
        self,
        action: CallToolAction,
        state: AgentState,
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> list[ObservationEvent]:

        if action.name not in self.tools:
            return [
                ToolFailed(
                    tool_name=action.name,
                    error_type="unknown_tool",
                    message=f"Unknown tool: {action.name}",
                )
            ]

        tool = self.tools[action.name]["fn"]

        try:
            result = await tool(**action.arguments)
            return [ToolProduced(tool_name=action.name, content=result)]
        except Exception as e:
            return [
                ToolFailed(
                    tool_name=action.name, error_type=type(e).__name__, message=str(e)
                )
            ]


class SimpleReducer(Reducer):
    async def apply(
        self,
        state: AgentState,
        events: Union[ObservationEvent, Sequence[ObservationEvent]],
    ) -> AgentState:
        # Normalize to list
        if not isinstance(events, Sequence):
            events = [events]

        for event in events:
            if isinstance(event, AssistantSaid):
                state.messages.append(Message(role="assistant", content=event.content))
                # Simple policy: stop if assistant replied with empty content
                if event.content.strip() == "":
                    state.done = True

            elif isinstance(event, ToolProduced):
                state.messages.append(
                    Message(
                        role="tool", name=event.tool_name, content=str(event.content)
                    )
                )

            elif isinstance(event, ToolFailed):
                state.messages.append(
                    Message(
                        role="tool",
                        name=event.tool_name,
                        content=f"[{event.error_type}] {event.message}",
                    )
                )
            else:
                # Unknown observation: no-op
                pass

        return state
