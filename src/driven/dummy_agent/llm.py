from collections import deque
from typing import AsyncIterator, Optional

from driven.harness.types import (
    LlmInput,
    LlmOutput,
    LlmStructuredResponse,
    LlmTextDelta,
    LlmTextResponse,
    LlmToolFunction,
    CallToolAction,
    FinishAction,
    Message,
    RespondAction,
    RunContext,
)


from driven.harness.protocols import (
    Controller,
    Emitter,
    Llm,
)


class ScriptedLlm(Llm):
    def __init__(
        self,
        responses: list[LlmOutput | LlmTextResponse | LlmStructuredResponse],
    ):
        self.responses = deque(responses)

        self.requests: list[LlmInput] = []
        self.tool_requests: list[tuple[LlmInput, list[LlmToolFunction]]] = []

    def _next_response(self):
        if not self.responses:
            raise RuntimeError("ScriptedLlm has no responses remaining")

        return self.responses.popleft()

    async def generate_text(
        self,
        request: LlmInput,
    ) -> LlmTextResponse:
        self.requests.append(request)

        response = self._next_response()

        if not isinstance(response, LlmTextResponse):
            raise TypeError(f"Expected LlmTextResponse, got {type(response)}")

        return response

    async def generate_structured(
        self,
        request: LlmInput,
        schema: dict,
    ) -> LlmStructuredResponse:
        self.requests.append(request)

        response = self._next_response()

        if not isinstance(response, LlmStructuredResponse):
            raise TypeError(f"Expected LlmStructuredResponse, got {type(response)}")

        return response

    async def chat_with_tools(
        self,
        request: LlmInput,
        tools: list[LlmToolFunction],
    ) -> LlmOutput:
        self.tool_requests.append((request, tools))

        response = self._next_response()

        if not isinstance(response, LlmOutput):
            raise TypeError(f"Expected LlmOutput, got {type(response)}")

        return response

    async def generate_text_stream(  # type: ignore
        self,
        request: LlmInput,
    ) -> AsyncIterator[LlmTextDelta]:
        response = await self.generate_text(request)

        yield LlmTextDelta(
            text=response.content,
            usage=response.usage,
        )


class LlmController(Controller):
    async def decide(
        self,
        run_context: RunContext,
        llm: Llm,
        sink: Optional[Emitter] = None,
    ):
        tools = run_context.get_tools_info()

        request = LlmInput(
            input=run_context.input,
            history=run_context.history,
            metadata=run_context.metadata,
        )

        response = await llm.chat_with_tools(
            request=request,
            tools=tools,
        )

        if response.tool_call:
            action = CallToolAction(
                tool_name=response.tool_call.name,
                arguments=response.tool_call.arguments,
            )

            message = Message(
                role="assistant",
                content="",
                metadata={
                    "tool_call": {
                        "name": response.tool_call.name,
                        "arguments": response.tool_call.arguments,
                    }
                },
            )

            return action, [message]

        if response.content:
            action = RespondAction(
                content=response.content,
            )

            message = Message(
                role="assistant",
                content=response.content,
            )

            return action, [message]

        return (
            FinishAction(content="done"),
            [],
        )
