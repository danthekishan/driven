import json
from typing import AsyncIterator, Optional

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.function_definition import (
    FunctionDefinition,
)

from driven.core.protocols import Llm
from driven.core.schemas import (
    LlmInput,
    LlmOutput,
    LlmStructuredResponse,
    LlmTextDelta,
    LlmTextResponse,
    LlmToolFunction,
    Message,
    ToolCall,
    Usage,
)


class OpenAILlm(Llm):
    def __init__(
        self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None
    ):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _convert_message(self, message: Message) -> ChatCompletionMessageParam:
        match message.role:
            case "system":
                return ChatCompletionSystemMessageParam(
                    role="system",
                    content=message.content,
                )

            case "user":
                return ChatCompletionUserMessageParam(
                    role="user",
                    content=message.content,
                )

            case "assistant":
                tool_calls = message.metadata.get("tool_calls", [])
                if tool_calls:
                    return ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=message.content,
                        tool_calls=[
                            {
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {
                                    "name": tc.get("name"),
                                    "arguments": json.dumps(tc.get("arguments", {})),
                                },
                            }
                            for tc in tool_calls
                        ],
                    )
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content,
                )

            case "tool":
                return ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=(message.tool_call_id or "unknown_tool_call"),
                    content=message.content,
                )

            case _:
                raise RuntimeError(f"unsupported message role: {message.role}")

    def _convert_messages(
        self,
        request: LlmInput,
    ) -> list[ChatCompletionMessageParam]:
        converted = [self._convert_message(msg) for msg in request.messages]
        if request.system:
            converted = [
                ChatCompletionSystemMessageParam(role="system", content=request.system),
                *converted,
            ]
        return converted

    def _convert_tools(
        self,
        tools: list[LlmToolFunction],
    ) -> list[ChatCompletionToolParam]:
        converted: list[ChatCompletionToolParam] = []

        for tool in tools:
            converted.append(
                ChatCompletionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters_schema,
                    ),
                )
            )

        return converted

    async def generate_text(self, request: LlmInput) -> LlmTextResponse:
        response: ChatCompletion = await self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(request),
        )

        message = response.choices[0].message

        return LlmTextResponse(
            content=message.content or "",
            stop_reason=response.choices[0].finish_reason,
            raw=response.model_dump(),
        )

    async def generate_structured(
        self, request: LlmInput, schema: dict
    ) -> LlmStructuredResponse:
        response: ChatCompletion = await self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(request),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                },
            },
        )

        message = response.choices[0].message
        content = message.content or "{}"

        return LlmStructuredResponse(
            data=json.loads(content),
            stop_reason=response.choices[0].finish_reason,
            raw=response.model_dump(),
        )

    async def chat_with_tools(
        self, request: LlmInput, tools: list[LlmToolFunction]
    ) -> LlmOutput:
        response: ChatCompletion = await self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(request),
            tools=self._convert_tools(tools),
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        usage = None
        if response.usage:
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        # tool calls
        if message.tool_calls:
            tool_calls: list[ToolCall] = []
            for tool_call in message.tool_calls:
                if tool_call.type != "function":
                    raise RuntimeError(f"unsupported tool type: {tool_call.type}")
                tool_calls.append(
                    ToolCall(
                        name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                        call_id=tool_call.id,
                    )
                )

            return LlmOutput(
                tool_calls=tool_calls,
                stop_reason=choice.finish_reason,
                usage=usage,
                raw=response.model_dump(),
            )

        # normal assistant text
        return LlmOutput(
            content=message.content or "",
            stop_reason=choice.finish_reason,
            usage=usage,
            raw=response.model_dump(),
        )

    async def generate_text_stream(  # type: ignore
        self, request: LlmInput
    ) -> AsyncIterator[LlmTextDelta]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(request),
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                yield LlmTextDelta(
                    text=delta.content,
                )
