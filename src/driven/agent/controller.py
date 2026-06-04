from dataclasses import dataclass, field
from typing import Callable, Optional

from driven.core.harness import (
    Controller,
    Emitter,
    Llm,
)
from driven.core.schemas import (
    ActionEvent,
    CallToolAction,
    LlmInput,
    LlmToolFunction,
    Message,
    RespondAction,
)


DEFAULT_INSTRUCTIONS = """
You are a helpful AI assistant.

You may use tools when needed.

Guidelines:
- Use tools only when necessary.
- Prefer direct answers when possible.
- If a tool fails, explain the failure clearly.
- Be concise but helpful.
- Think step-by-step before using tools.
- When the task is complete, respond normally.
"""


@dataclass
class ToolCallingController(Controller):
    instructions: str = DEFAULT_INSTRUCTIONS

    append_system_message: bool = True

    metadata: dict = field(default_factory=dict)

    async def decide(
        self,
        messages: list[Message],
        get_tools: Callable[..., list[LlmToolFunction]],
        llm: Llm,
        emitter: Optional[Emitter] = None,
    ) -> list[ActionEvent]:
        tools = get_tools()

        request_messages = self._build_messages(messages)

        request = LlmInput(
            messages=request_messages,
            metadata=self.metadata,
        )

        if emitter:
            await emitter.emit(
                "controller.request",
                {
                    "message_count": (len(request_messages)),
                    "tool_count": len(tools),
                },
            )

        response = await llm.chat_with_tools(
            request=request,
            tools=tools,
        )

        if emitter:
            await emitter.emit(
                "controller.response",
                {
                    "has_tool_call": (response.tool_call is not None),
                    "has_content": (response.content is not None),
                    "stop_reason": (response.stop_reason),
                },
            )

        return self._convert_response_to_actions(
            response=response,
            emitter=emitter,
        )

    # =========================
    # INTERNALS
    # =========================

    def _build_messages(
        self,
        messages: list[Message],
    ) -> list[Message]:
        if not self.append_system_message:
            return messages

        system_message = Message(
            role="system",
            content=self.instructions,
        )

        return [
            system_message,
            *messages,
        ]

    def _convert_response_to_actions(
        self,
        response,
        emitter: Optional[Emitter] = None,
    ) -> list[ActionEvent]:
        actions: list[ActionEvent] = []

        # tool call
        if response.tool_call:
            if emitter:
                # fire and forget style
                # controller should not fail
                # because telemetry failed
                pass

            actions.append(
                CallToolAction(
                    name=response.tool_call.name,
                    arguments=(response.tool_call.arguments),
                )
            )

        # assistant response
        if response.content:
            actions.append(RespondAction(content=response.content))

        # fallback
        if not actions:
            actions.append(RespondAction(content=("No response generated.")))

        return actions
