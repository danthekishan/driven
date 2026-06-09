import inspect
from typing import Any, Callable, Optional, Type, get_type_hints

from pydantic import BaseModel, create_model

from driven.core.schemas import JSONValue, LlmToolFunction


RUNTIME_PARAMS = {"state", "llm", "emitter", "spawn_branch"}


class ToolDescriptor:
    __slots__ = ("_func", "_name", "_description", "_public", "__name__", "__qualname__")

    def __init__(self, func: Callable, name: str = "", description: str = "", public: bool = False):
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or "No description provided."
        self._public = public
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __get__(self, obj, objtype=None):
        return ToolDescriptor(
            self._func.__get__(obj, objtype),
            name=self._name,
            description=self._description,
            public=self._public,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def public(self) -> bool:
        return self._public


def tool(*, name: str = "", description: str = "", public: bool = False):
    def decorator(f):
        return ToolDescriptor(f, name=name, description=description, public=public)
    return decorator


class Tool:
    __slots__ = ("func", "name", "description", "args_schema", "runtime_params", "is_async", "pydantic_param")

    def __init__(
        self,
        func: Callable,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        runtime_params: set[str],
        pydantic_param: Optional[str] = None,
    ):
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.runtime_params = runtime_params
        self.is_async = inspect.iscoroutinefunction(func)
        self.pydantic_param = pydantic_param

    async def run(self, args: dict[str, Any], runtime_context: dict | None = None) -> JSONValue:
        runtime_context = runtime_context or {}
        validated = self.args_schema.model_validate(args)
        runtime_kwargs = {p: runtime_context.get(p) for p in self.runtime_params}

        if self.pydantic_param:
            kwargs = {self.pydantic_param: validated, **runtime_kwargs}
        else:
            kwargs = validated.model_dump()
            kwargs.update(runtime_kwargs)

        if self.is_async:
            return await self.func(**kwargs)
        return self.func(**kwargs)

    def get_schema(self, full_name: str) -> LlmToolFunction:
        return LlmToolFunction(
            name=full_name,
            description=self.description,
            parameters_schema=self.args_schema.model_json_schema(),
        )


class RegisteredTool:
    __slots__ = ("extension_name", "tool")

    def __init__(self, extension_name: str, tool: Tool):
        self.extension_name = extension_name
        self.tool = tool


def build_tool(func: Callable, name: str, description: str) -> Tool:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    runtime_params: set[str] = set()
    fields: dict[str, Any] = {}

    for param in sig.parameters.values():
        if param.name in ("self", "cls"):
            continue
        if param.name in RUNTIME_PARAMS:
            runtime_params.add(param.name)
            continue
        param_type = type_hints.get(param.name, Any)
        if param.default == inspect.Parameter.empty:
            fields[param.name] = (param_type, ...)
        else:
            fields[param.name] = (param_type, param.default)

    if not fields:
        return Tool(func, name, description, create_model(f"{name}InputSchema"), runtime_params)

    if len(fields) == 1:
        first_name, (first_type, _) = next(iter(fields.items()))
        if inspect.isclass(first_type) and issubclass(first_type, BaseModel):
            return Tool(func, name, description, first_type, runtime_params, pydantic_param=first_name)

    return Tool(func, name, description, create_model(f"{name}InputSchema", **fields), runtime_params)


def discover_tools(instance) -> list[tuple[Tool, bool]]:
    results: list[tuple[Tool, bool]] = []
    for _, attr in inspect.getmembers(
        instance, predicate=lambda m: isinstance(m, ToolDescriptor)
    ):
        tool_obj = build_tool(attr._func, attr.name, attr.description)
        results.append((tool_obj, attr.public))
    return results
