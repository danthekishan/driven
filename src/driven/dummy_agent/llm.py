from collections import deque
from typing import AsyncIterator, Callable, Optional

from driven.harness.types import (
    ActionEvent,
    LlmInput,
    LlmOutput,
    LlmStructuredResponse,
    LlmTextDelta,
    LlmTextResponse,
    LlmToolFunction,
    CallToolAction,
    Message,
    RespondAction,
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
        messages: list[Message],
        get_tools: Callable[..., list[LlmToolFunction]],
        llm: Llm,
        sink: Optional[Emitter] = None,
    ) -> list[ActionEvent]:
        request = LlmInput(messages=messages)
        tools = get_tools()
        response = await llm.chat_with_tools(request=request, tools=tools)

        if response.tool_call:
            return [
                CallToolAction(
                    name=response.tool_call.name, arguments=response.tool_call.arguments
                )
            ]

        if response.content is not None:
            return [RespondAction(content=response.content)]

        # Nothing useful: return an empty assistant response; reducer policy may stop after this
        return [RespondAction(content="")]
