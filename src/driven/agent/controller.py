from dataclasses import dataclass
from typing import AsyncIterator, Callable

from driven.core.harness import CallTools, Controller, Llm
from driven.core.schemas import (
    AssistantFinalized,
    ControllerRequested,
    LlmInput,
    LlmOutputReceived,
    LlmRequestPrepared,
    LlmToolFunction,
    Message,
    ToolsCompleted,
    ToolsRequested,
    TurnEvent,
    TurnFinished,
)


@dataclass
class ToolCallingController(Controller):
    async def stream_turn(
        self,
        messages: list[Message],
        system: str,
        get_tools: Callable[..., list[LlmToolFunction]],
        call_tools: CallTools,
        llm: Llm,
    ) -> AsyncIterator[TurnEvent]:
        tools = get_tools()
        working_messages = list(messages)

        while True:
            yield ControllerRequested(
                message_count=len(working_messages),
                tool_count=len(tools),
            )

            request = LlmInput(system=system, messages=working_messages)
            yield LlmRequestPrepared(request=request, tool_count=len(tools))

            output = await llm.chat_with_tools(request=request, tools=tools)
            yield LlmOutputReceived(output=output)

            tool_calls = list(output.tool_calls)

            if tool_calls:
                yield ToolsRequested(
                    tool_calls=tool_calls,
                    raw_llm_response=output.raw,
                )

                results = await call_tools(tool_calls, None)
                yield ToolsCompleted(results=results, raw_llm_response=output.raw)

                working_messages.append(
                    Message(
                        role="assistant",
                        content="",
                        metadata={
                            "tool_calls": [
                                {
                                    "id": tc.call_id,
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }
                                for tc in tool_calls
                            ]
                        },
                    )
                )

                for result in results:
                    if result.ok:
                        content = str(result.content)
                    else:
                        content = f"[{result.error_type}] {result.error_message}"
                    working_messages.append(
                        Message(
                            role="tool",
                            name=result.name,
                            tool_call_id=result.call_id,
                            content=content,
                        )
                    )

                continue

            yield AssistantFinalized(
                content=output.content or "No response generated.",
                stop_reason=output.stop_reason,
                usage=output.usage,
                raw_llm_response=output.raw,
            )
            yield TurnFinished(done=True, reason="assistant_response")
            return
