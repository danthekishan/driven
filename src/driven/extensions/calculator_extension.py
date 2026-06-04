from driven.core.tool_runtime import Extension, tool


class MathExtension(Extension):
    default_id = "math_expert"
    instructions = "You are a strict math assistant. You only return precise numbers."

    @tool(description="Adds two numbers together.")
    def add(self, a: float, b: float) -> float:
        return a + b

    @tool(description="Multiplies two numbers. Yields to event loop for heavy compute.")
    async def multiply(self, a: float, b: float) -> float:
        import asyncio

        await asyncio.sleep(0.1)  # Simulate heavy async work
        return a * b
