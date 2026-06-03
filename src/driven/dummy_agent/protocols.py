from typing import Any, Optional, Sequence, Union
from driven.harness.protocols import StateManager, Emitter, Llm, Runtime
from driven.harness.types import (
    AgentState,
    Message,
    CallToolAction,
    AssistantResp,
    ObservationEvent,
    ToolProduced,
    ToolFailed,
    LlmToolFunction,
)


class InMemoryStateManager(StateManager):
    def __init__(self):
        self.store: dict[str, AgentState] = {}

    async def load(self, run_id: str) -> AgentState:
        if run_id in self.store:
            return self.store[run_id]

        state = AgentState(state_id=run_id, prompt="", messages=[])
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
