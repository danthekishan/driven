import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, get_type_hints

from pydantic import BaseModel, create_model, Field

from driven.core.harness import Emitter, HarnessState, Llm
from driven.core.schemas import (
    JSONValue,
    LlmToolFunction,
    ToolCall,
    ToolResult,
)


RUNTIME_PARAMS = {"state", "llm", "emitter", "spawn_branch"}


class Tool(BaseModel):
    func: Callable
    name: str
    description: str
    args_schema: Type[BaseModel]
    is_input_pydantic: bool
    is_async: bool

    runtime_params: set[str] = Field(default_factory=set)

    async def run(self, args: dict[str, Any], runtime_context: dict = {}) -> JSONValue:
        validated_args = self.args_schema.model_validate(args)
        kwargs = validated_args.model_dump()

        for param_name in self.runtime_params:
            kwargs[param_name] = runtime_context.get(param_name)

        match (self.is_async, self.is_input_pydantic):
            case True, True:
                return await self.func(validated_args)
            case True, False:
                return await self.func(**kwargs)
            case False, True:
                return self.func(validated_args)
            case False, False:
                return self.func(**kwargs)

    def get_schema(self, full_name: str) -> LlmToolFunction:
        return LlmToolFunction(
            name=full_name,
            description=self.description,
            parameters_schema=(self.args_schema.model_json_schema()),
        )


@dataclass(slots=True)
class RegisteredTool:
    extension_name: str
    tool: Tool


def build_tool(func: Callable, metadata: dict[str, Any]) -> Tool:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    is_async = inspect.iscoroutinefunction(func)
    runtime_params: set[str] = set()
    fields: dict[str, Any] = {}
    params = [
        param for param in sig.parameters.values() if param.name not in ("self", "cls")
    ]

    for param in params:
        if param.name in RUNTIME_PARAMS:
            runtime_params.add(param.name)
            continue

        param_type = type_hints.get(param.name, Any)
        if param.default == inspect.Parameter.empty:
            fields[param.name] = (param_type, ...)
        else:
            fields[param.name] = (param_type, param.default)

    if len(fields) == 0:
        args_schema = create_model(f"{metadata['name']}InputSchema")
        return Tool(
            func=func,
            name=metadata["name"],
            description=metadata["description"],
            args_schema=args_schema,
            is_input_pydantic=False,
            is_async=is_async,
            runtime_params=runtime_params,
        )

    if len(fields) == 1:
        first_param_type = list(fields.values())[0][0]
        if inspect.isclass(first_param_type) and issubclass(
            first_param_type, BaseModel
        ):
            return Tool(
                func=func,
                name=metadata["name"],
                description=metadata["description"],
                args_schema=(first_param_type),
                is_input_pydantic=True,
                is_async=is_async,
                runtime_params=runtime_params,
            )

    args_schema = create_model(f"{metadata['name']}InputSchema", **fields)
    return Tool(
        func=func,
        name=metadata["name"],
        description=metadata["description"],
        args_schema=args_schema,
        is_input_pydantic=False,
        is_async=is_async,
        runtime_params=runtime_params,
    )


def tool(name: Optional[str] = None, description: Optional[str] = None):
    def decorator(func):
        setattr(
            func,
            "__tool_metadata__",
            {
                "name": (name or func.__name__),
                "description": (
                    description or func.__doc__ or "No description provided."
                ),
            },
        )
        return func

    return decorator


class Extension:
    name: str = "extension"
    version: str = "0.0.1"
    description: str = ""
    instructions: str = "Generic extension."
    enabled: bool = True
    requires: list[str] = []

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._discover_tools()

    def _discover_tools(self):
        for _, method in inspect.getmembers(self, predicate=inspect.ismethod):
            metadata = getattr(method, "__tool_metadata__", None)

            if metadata is None:
                continue

            tool_obj = build_tool(func=method, metadata=metadata)
            self.tools[tool_obj.name] = tool_obj

    async def start(self):
        pass

    async def stop(self):
        pass


class ExtensionRegistry:
    def __init__(
        self,
        exts: list[Extension] = [],
        max_start_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        self.exts = exts
        self.extensions: dict[str, Extension] = {}
        self.tools: dict[str, RegisteredTool] = {}
        self.max_start_attempts = max_start_attempts
        self.retry_delay = retry_delay
        self._tg: asyncio.TaskGroup | None = None

    async def __aenter__(self):
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()

        for ext in self.exts:
            await self.register(ext)

        return self

    async def __aexit__(self, exc_type, exc, tb):
        for extension in self.extensions.values():
            try:
                await extension.stop()
            except Exception:
                pass
        assert self._tg

        await self._tg.__aexit__(exc_type, exc, tb)

    async def _start_extension(self, extension: Extension):
        last_exception: Exception | None = None

        for attempt in range(1, self.max_start_attempts + 1):
            try:
                await extension.start()
                return

            except asyncio.CancelledError:
                raise

            except Exception as e:
                last_exception = e
                print(
                    f"[registry] failed to start "
                    f"extension='{extension.name}' "
                    f"attempt={attempt}/"
                    f"{self.max_start_attempts} "
                    f"error={e}"
                )

                try:
                    await extension.stop()
                except Exception:
                    pass

                if attempt < self.max_start_attempts:
                    await asyncio.sleep(self.retry_delay)

        raise RuntimeError(
            f"Failed to start extension "
            f"'{extension.name}' "
            f"after "
            f"{self.max_start_attempts} "
            f"attempts"
        ) from last_exception

    async def register(self, extension: Extension):
        if extension.name in self.extensions:
            raise RuntimeError(f"Extension '{extension.name}' already registered")

        await self._start_extension(extension)
        self.extensions[extension.name] = extension

        for local_name, tool_obj in extension.tools.items():
            full_name = f"{extension.name}-{local_name}"

            if full_name in self.tools:
                raise RuntimeError(f"Duplicate tool '{full_name}'")

            self.tools[full_name] = RegisteredTool(
                extension_name=(extension.name), tool=tool_obj
            )

    def get_tools(
        self, extension_name: (Optional[str]) = None
    ) -> list[LlmToolFunction]:
        tools: list[LlmToolFunction] = []
        for full_name, registered in self.tools.items():
            if extension_name and registered.extension_name != extension_name:
                continue
            tools.append(registered.tool.get_schema(full_name=full_name))
        return tools

    async def call_tools(
        self,
        tools: list[ToolCall],
        state: Optional[HarnessState],
        llm: Optional[Llm],
        emitter: (Optional[Emitter]) = None,
        timeout: Optional[float] = None,
        spawn_branch: Optional[Callable] = None,
    ) -> list[ToolResult]:
        async def _run_one(call: ToolCall) -> ToolResult:
            try:
                registered = self.tools.get(call.name)
                if registered is None:
                    raise RuntimeError(f"Unknown tool '{call.name}'")

                result = await registered.tool.run(
                    call.arguments,
                    runtime_context={
                        "state": state,
                        "llm": llm,
                        "emitter": emitter,
                        "spawn_branch": spawn_branch,
                    },
                )
                return ToolResult(
                    name=call.name,
                    ok=True,
                    content=result,
                    call_id=call.call_id,
                )
            except Exception as e:
                return ToolResult(
                    name=call.name,
                    ok=False,
                    call_id=call.call_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )

        try:
            async with asyncio.timeout(timeout):
                return await asyncio.gather(*[_run_one(call) for call in tools])
        except TimeoutError:
            return [
                ToolResult(
                    name=call.name,
                    ok=False,
                    call_id=call.call_id,
                    error_type="TimeoutError",
                    error_message="tool batch timed out",
                )
                for call in tools
            ]
