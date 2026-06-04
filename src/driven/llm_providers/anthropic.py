import json
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolParam
from anthropic.types.message import Message as AnthropicMessage

from driven.core.harness import Llm
from driven.core.schemas import (
    LlmInput,
    LlmOutput,
    LlmStructuredResponse,
    LlmTextDelta,
    LlmTextResponse,
    LlmToolFunction,
    Message,
    ToolCall,
)


class AnthropicLlm(Llm):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)

    def _extract_system_message(
        self, messages: list[Message]
    ) -> tuple[Optional[str], list[Message]]:
        system_parts: list[str] = []
        non_system: list[Message] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                non_system.append(msg)

        system = "\n\n".join(system_parts) or None

        return system, non_system

    def _convert_message(
        self,
        message: Message,
    ) -> MessageParam:
        match message.role:
            case "user":
                return {
                    "role": "user",
                    "content": message.content,
                }

            case "assistant":
                return {
                    "role": "assistant",
                    "content": message.content,
                }

            case "tool":
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": (
                                message.tool_call_id or "unknown_tool_call"
                            ),
                            "content": message.content,
                        }
                    ],
                }

            case "system":
                raise RuntimeError(
                    "system messages must be extracted separately for anthropic"
                )

            case _:
                raise RuntimeError(f"unsupported role: {message.role}")

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[Optional[str], list[MessageParam]]:
        system, non_system = self._extract_system_message(messages)
        converted = [self._convert_message(msg) for msg in non_system]
        return system, converted

    def _convert_tools(self, tools: list[LlmToolFunction]) -> list[ToolParam]:
        converted: list[ToolParam] = []

        for tool in tools:
            converted.append(
                ToolParam(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.parameters_schema,
                )
            )

        return converted

    def _extract_text(self, response: AnthropicMessage) -> str:
        parts: list[str] = []

        for block in response.content:
            if block.type == "text":
                parts.append(block.text)

        return "".join(parts)

    async def generate_text(self, request: LlmInput) -> LlmTextResponse:
        system, messages = self._convert_messages(request.messages)
        response: AnthropicMessage = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=messages,
        )

        return LlmTextResponse(
            content=self._extract_text(response),
            stop_reason=response.stop_reason,
            raw=response.model_dump(),
        )

    async def generate_structured(
        self, request: LlmInput, schema: dict
    ) -> LlmStructuredResponse:
        system, messages = self._convert_messages(request.messages)

        schema_instruction = (
            "You must respond with valid JSON "
            "matching this schema:\n\n"
            f"{json.dumps(schema, indent=2)}"
        )

        if system:
            system = f"{system}\n\n{schema_instruction}"
        else:
            system = schema_instruction

        response: AnthropicMessage = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )

        text = self._extract_text(response)

        return LlmStructuredResponse(
            data=json.loads(text),
            stop_reason=response.stop_reason,
            raw=response.model_dump(),
        )

    async def chat_with_tools(
        self, request: LlmInput, tools: list[LlmToolFunction]
    ) -> LlmOutput:
        system, messages = self._convert_messages(request.messages)
        response: AnthropicMessage = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=messages,
            tools=self._convert_tools(tools),
        )

        # anthropic content blocks
        for block in response.content:
            if block.type == "tool_use":
                return LlmOutput(
                    tool_call=ToolCall(
                        name=block.name,
                        arguments=(
                            block.input if isinstance(block.input, dict) else {}
                        ),
                        call_id=block.id,
                    ),
                    stop_reason=response.stop_reason,
                    raw=response.model_dump(),
                )

        # normal text response
        return LlmOutput(
            content=self._extract_text(response),
            stop_reason=response.stop_reason,
            raw=response.model_dump(),
        )

    async def generate_text_stream(  # type: ignore
        self, request: LlmInput
    ) -> AsyncIterator[LlmTextDelta]:
        system, messages = self._convert_messages(request.messages)

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or "",
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield LlmTextDelta(
                    text=text,
                )
