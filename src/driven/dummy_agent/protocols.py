from typing import Any, Optional
from driven.harness.protocols import StateManager, Emitter, Llm, Runtime, Reducer
from driven.harness.types import (
    AgentState,
    LlmRequestEventPayload,
    LlmResponseEventPayload,
    Message,
    RunContext,
    AgentEvent,
    Action,
    CallToolAction,
    EventType,
    FinishAction,
    Observation,
    RespondAction,
    ToolResult,
    LlmToolFunction,
    ObservationEventPayload,
    GenericEventPayload,
    ActionEventPayload,
)


class InMemoryStateManager(StateManager):
    def __init__(self):
        self.store: dict[str, AgentState] = {}

    async def load(
        self,
        run_id: str,
        input: list[Message],
    ) -> AgentState:
        if run_id in self.store:
            return self.store[run_id]

        state = AgentState(
            state_id=run_id,
            run_context=RunContext(
                id=run_id,
                state_id=run_id,
                history=input,
                metadata={},
                get_tools_info=lambda: [],
            ),
            history=input.copy(),
        )

        self.store[run_id] = state

        return state

    async def save(
        self,
        state: AgentState,
    ) -> AgentState:
        self.store[state.state_id] = state

        return state


class PrintEmitter(Emitter):
    async def emit(
        self,
        event: AgentEvent,
    ) -> None:
        print(event)


class DummyRuntime(Runtime):
    def __init__(self):
        self.tools: dict[str, Any] = {}

    def register_tool(
        self,
        name: str,
        fn,
        description: str = "",
    ):
        self.tools[name] = {
            "fn": fn,
            "description": description,
        }

    def get_tools(
        self,
        *args,
        **kwargs,
    ) -> list[LlmToolFunction]:
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
        action: Action,
        state: AgentState,
        run_context: RunContext,
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> tuple[Observation, list[Message]]:

        if isinstance(action, RespondAction):
            observation = Observation(
                data=action.content,
                kind="tool_result",
            )

            return observation, []

        if isinstance(action, FinishAction):
            observation = Observation(
                data=action.content,
                kind="tool_result",
            )

            return observation, []

        if isinstance(action, CallToolAction):
            if action.tool_name not in self.tools:
                raise RuntimeError(f"Unknown tool: {action.tool_name}")

            tool = self.tools[action.tool_name]["fn"]

            if sink:
                await sink.emit(
                    # TODO:
                    AgentEvent(
                        payload=LlmRequestEventPayload(
                            type=EventType.TOOL_CALL_STARTED,
                            op_type="tool",
                            model="xxxx",
                            prompt="",
                            extra={
                                "tool": action.tool_name,
                            },
                        )
                    )
                )

            result = await tool(**action.arguments)

            if sink:
                await sink.emit(
                    # TODO:
                    AgentEvent(
                        payload=LlmResponseEventPayload(
                            type=EventType.TOOL_CALL_ENDED,
                            op_type="tool",
                            model="xxxx",
                            prompt="",
                            response={},
                            extra={
                                "tool": action.tool_name,
                            },
                        )
                    )
                )

            observation = Observation(
                data=ToolResult(
                    ok=True,
                    content=result,
                ),
                kind="tool_result",
            )

            tool_message = Message(
                role="tool",
                content=str(result),
                name=action.tool_name,
            )

            return observation, [tool_message]

        raise RuntimeError(f"Unhandled action type: {type(action)}")


class SimpleReducer(Reducer):
    async def apply(
        self,
        state: AgentState,
        event: AgentEvent,
    ) -> AgentState:

        payload = event.payload

        match payload:
            case GenericEventPayload(type=EventType.STEP_STARTED):
                state.step += 1

            case ActionEventPayload():
                state.last_action = payload.action
                state.history.extend(payload.agent_messages)

                if isinstance(
                    payload.action,
                    FinishAction,
                ):
                    state.done = True

            case ObservationEventPayload():
                state.last_observation = payload.observation

                state.history.extend(payload.tool_messages)

        return state
