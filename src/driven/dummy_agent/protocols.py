from typing import Any, Optional
from driven.core.harness import HarnessState, StateManager, Emitter, Llm, Runtime
from driven.core.schemas import (
    JSONValue,
    ObservationEvent,
    ToolProduced,
    ToolFailed,
    LlmToolFunction,
)


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
    async def emit(self, event_type, events) -> None:
        # Accept single or list
        if isinstance(events, (list, tuple)):
            for e in events:
                print(event_type, e)
        else:
            print(event_type, events)


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

    async def call_tool(
        self,
        fn_name: str,
        arguments: dict[str, JSONValue],
        state: HarnessState,
        llm: Llm,
        emitter: Optional[Emitter] = None,
        timeout: Optional[float] = None,
    ) -> list[ObservationEvent]:

        if fn_name not in self.tools:
            return [
                ToolFailed(
                    tool_name=fn_name,
                    error_type="unknown_tool",
                    message=f"Unknown tool: {fn_name}",
                )
            ]

        tool = self.tools[fn_name]["fn"]

        try:
            result = await tool(**arguments)
            return [ToolProduced(tool_name=fn_name, content=result)]
        except Exception as e:
            return [
                ToolFailed(
                    tool_name=fn_name, error_type=type(e).__name__, message=str(e)
                )
            ]
