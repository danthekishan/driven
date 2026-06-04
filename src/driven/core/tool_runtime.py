# import asyncio
# import inspect
# from typing import Any, Callable, Literal, Optional, Type, get_type_hints
# from pydantic import BaseModel, create_model, Field
#
# from driven.core.harness import Emitter, HarnessState, Llm
# from driven.core.types import (
#     JSONValue,
#     LlmToolFunction,
#     ObservationEvent,
#     ToolFailed,
#     ToolProduced,
# )
#
#
# class Tool(BaseModel):
#     func: Callable
#     name: str
#     description: str
#     args_schema: Type[BaseModel]
#     is_input_pydantic: bool
#     is_async: bool
#
#     async def run(self, args: dict[str, Any]) -> JSONValue:
#         validated_args = self.args_schema.model_validate(args)
#
#         match self.is_async, self.is_input_pydantic:
#             case True, True:
#                 return await self.func(validated_args)
#             case True, False:
#                 return await self.func(**validated_args.model_dump())
#             case False, True:
#                 return self.func(validated_args)
#             case False, False:
#                 return self.func(**validated_args.model_dump())
#
#     def get_json_schema(self) -> dict[str, Any]:
#         """Returns a clean JSON schema optimized for LLMs or API frameworks."""
#         return {
#             "name": self.name,
#             "description": self.description,
#             "parameters": self.args_schema.model_json_schema(),
#         }
#
#     def get_schema(self) -> LlmToolFunction:
#         """Returns a clean JSON schema optimized for LLMs or API frameworks."""
#         return LlmToolFunction(
#             name=self.name,
#             description=self.description,
#             parameters_schema=self.args_schema.model_json_schema(),
#         )
#
#
# def build_tool(func: Callable, metadata: dict[str, Any]) -> Tool:
#     tool_name = metadata["name"]
#     tool_description = metadata["description"]
#
#     sig = inspect.signature(func)
#     type_hints = get_type_hints(func)
#     is_async = inspect.iscoroutinefunction(func)
#
#     params = [
#         param for param in sig.parameters.values() if param.name not in ("self", "cls")
#     ]
#
#     # Pydantic Input
#     if len(params) == 1:
#         param_type = type_hints.get(params[0].name)
#         if (
#             param_type
#             and inspect.isclass(param_type)
#             and issubclass(param_type, BaseModel)
#         ):
#             return Tool(
#                 func=func,
#                 name=tool_name,
#                 description=tool_description,
#                 args_schema=param_type,
#                 is_input_pydantic=True,
#                 is_async=is_async,
#             )
#
#     # Auto Generated Schema
#     fields: dict[str, Any] = {}
#
#     for param in params:
#         param_type = type_hints.get(param.name, Any)
#
#         if param.default == inspect.Parameter.empty:
#             fields[param.name] = (param_type, ...)
#         else:
#             fields[param.name] = (
#                 param_type,
#                 param.default,
#             )
#
#     args_schema = create_model(
#         f"{tool_name}InputSchema",
#         **fields,
#     )
#
#     return Tool(
#         func=func,
#         name=tool_name,
#         description=tool_description,
#         args_schema=args_schema,
#         is_input_pydantic=False,
#         is_async=is_async,
#     )
#
#
# def tool(name=None, description=None):
#
#     def decorator(func):
#
#         setattr(
#             func,
#             "__tool_metadata__",
#             {
#                 "name": name or func.__name__,
#                 "description": (
#                     description or func.__doc__ or "No description provided."
#                 ),
#             },
#         )
#
#         return func
#
#     return decorator
#
#
# type ExtensionRequestType = Literal["info", "tool_call"]
#
#
# class ExtensionRequest(BaseModel):
#     sender_id: str
#     request_type: ExtensionRequestType
#     request_body: dict[str, Any] = Field(default_factory=dict)
#     request_config: dict[str, Any] = Field(default_factory=dict)
#     request_context: dict[str, Any] = Field(default_factory=dict)
#     reply_to: Optional[asyncio.Future] = None
#
#
# class Extension:
#     """
#     Base extension class.
#     """
#
#     name: str = "extension"
#     version: str = "0.0.1"
#     description: str = ""
#     instructions: str = "Generic extension."
#     requires: list[str] = []
#     enabled: bool = True
#
#     def __init__(self):
#         self.tools: dict[str, Tool] = {}
#         self.mailbox: asyncio.Queue[Optional[ExtensionRequest]] = asyncio.Queue()
#         self.tasks: set[asyncio.Task] = set()
#         self._discover_tools()
#
#     def _discover_tools(self):
#
#         for _, method in inspect.getmembers(
#             self,
#             predicate=inspect.ismethod,
#         ):
#             metadata = getattr(method, "__tool_metadata__", None)
#
#             if metadata is None:
#                 continue
#
#             tool_obj = build_tool(func=method, metadata=metadata)
#             namespaced_name = f"{self.name}.{tool_obj.name}"
#             tool_obj.name = namespaced_name
#             self.tools[namespaced_name] = tool_obj
#
#     async def setup(self):
#         pass
#
#     async def teardown(self):
#         pass
#
#     async def handle(self, request: ExtensionRequest) -> JSONValue:
#
#         match request.request_type:
#             case "info":
#                 return {
#                     "name": self.name,
#                     "version": self.version,
#                     "description": self.description,
#                     "instructions": self.instructions,
#                     "tools": [tool.get_json_schema() for tool in self.tools.values()],
#                 }
#
#             case "tool_call":
#                 tool_name = request.request_body["tool_name"]
#                 args = request.request_body.get("args", {})
#                 tool_obj = self.tools.get(tool_name)
#
#                 if tool_obj is None:
#                     raise RuntimeError(f"Tool '{tool_name}' not found")
#
#                 result = await tool_obj.run(args)
#                 return result
#
#         raise RuntimeError(f"Unknown request type: {request.request_type}")
#
#     async def _run_loop(self):
#         await self.setup()
#         try:
#             while True:
#                 msg = await self.mailbox.get()
#                 if msg is None:
#                     break
#                 asyncio.create_task(self._process_and_reply(msg))
#         finally:
#             for task in self.tasks:
#                 task.cancel()
#             await self.teardown()
#
#     async def _process_and_reply(self, msg: ExtensionRequest):
#         try:
#             result = await self.handle(msg)
#             if msg.reply_to and not msg.reply_to.done():
#                 msg.reply_to.set_result(result)
#         except Exception as e:
#             if msg.reply_to and not msg.reply_to.done():
#                 msg.reply_to.set_exception(e)
#
#
# class ExtensionRegistry:
#     def __init__(self, config: Optional[dict] = None):
#         self.extensions: dict[str, Extension] = {}
#         self.mailboxes: dict[str, asyncio.Queue] = {}
#         self.tools: dict[str, Tool] = {}
#         self.config = config or {}
#         self.state: dict[str, Any] = {}
#         self._tg: Optional[asyncio.TaskGroup] = None
#
#     async def __aenter__(self):
#         self._tg = asyncio.TaskGroup()
#         await self._tg.__aenter__()
#         return self
#
#     async def __aexit__(self, exc_type, exc, tb):
#         for mailbox in self.mailboxes.values():
#             await mailbox.put(None)
#         assert self._tg
#         await self._tg.__aexit__(exc_type, exc, tb)
#
#     def register(self, extension: Extension):
#         if self._tg is None:
#             raise RuntimeError("Registry must be used within async with")
#
#         if extension.name in self.extensions:
#             raise RuntimeError(f"Extension '{extension.name}' already exists")
#
#         self.extensions[extension.name] = extension
#         self.mailboxes[extension.name] = extension.mailbox
#
#         for tool_name, tool in extension.tools.items():
#             if tool_name in self.tools:
#                 raise RuntimeError(f"Duplicate tool '{tool_name}'")
#             self.tools[tool_name] = tool
#
#         self._tg.create_task(extension._run_loop())
#
#     async def ask(
#         self,
#         target: str,
#         sender_id: str,
#         request_type: ExtensionRequestType,
#         body: dict,
#         config: dict,
#         context: dict,
#         timeout: Optional[float] = None,
#     ):
#         if target not in self.mailboxes:
#             raise RuntimeError(f"Unknown extension '{target}'")
#         reply_future = asyncio.get_running_loop().create_future()
#         msg = ExtensionRequest(
#             sender_id=sender_id,
#             request_type=request_type,
#             request_body=body,
#             request_config=config,
#             request_context=context,
#             reply_to=reply_future,
#         )
#         await self.mailboxes[target].put(msg)
#         async with asyncio.timeout(timeout):
#             return await reply_future
#
#     def get_tools(self, *args, extension_name: str) -> list[LlmToolFunction]:
#         extension = self.extensions.get(extension_name)
#
#         if not extension:
#             raise ValueError(f"Extension: {extension_name} is not found")
#
#         return [tool.get_schema() for tool in extension.tools.values()]
#
#     async def call_tool(
#         self,
#         fn_name: str,
#         arguments: dict[str, JSONValue],
#         state: HarnessState,
#         llm: Llm,
#         emitter: Optional[Emitter] = None,
#         timeout: Optional[float] = None,
#     ) -> list[ObservationEvent]:
#         try:
#             if fn_name not in self.tools:
#                 raise RuntimeError(f"Unknown tool '{fn_name}'")
#
#             extension_name = fn_name.split(".")[0]
#             result = await self.ask(
#                 target=extension_name,
#                 sender_id="registry",
#                 request_type="tool_call",
#                 body={
#                     "tool_name": fn_name,
#                     "args": arguments,
#                 },
#                 context={
#                     "state": state,
#                     "llm": llm,
#                     "emitter": emitter,
#                 },
#                 config={},
#                 timeout=timeout,
#             )
#             return [ToolProduced(tool_name=fn_name, content=result)]
#         except Exception as e:
#             return [
#                 ToolFailed(
#                     tool_name=fn_name,
#                     error_type=type(e).__name__,
#                     message=str(e),
#                 )
#             ]


import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, get_type_hints

from pydantic import BaseModel, create_model, Field

from driven.core.harness import Emitter, HarnessState, Llm
from driven.core.schemas import (
    JSONValue,
    LlmToolFunction,
    ObservationEvent,
    ToolFailed,
    ToolProduced,
)


RUNTIME_PARAMS = {"state", "llm", "emitter"}


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

    # build visible fields
    for param in params:
        if param.name in RUNTIME_PARAMS:
            runtime_params.add(param.name)
            continue

        param_type = type_hints.get(param.name, Any)
        if param.default == inspect.Parameter.empty:
            fields[param.name] = (param_type, ...)
        else:
            fields[param.name] = (param_type, param.default)

    # no input params
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

    # pydantic input mode
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

    # generated schema
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
        """
        Setup resources.

        Examples:
            - database connections
            - websocket sessions
            - background workers
            - API clients
        """
        pass

    async def stop(self):
        """
        Cleanup resources.
        """
        pass


class ExtensionRegistry:
    def __init__(self, max_start_attempts: int = 3, retry_delay: float = 2.0):
        self.extensions: dict[str, Extension] = {}
        self.tools: dict[str, RegisteredTool] = {}
        self.max_start_attempts = max_start_attempts
        self.retry_delay = retry_delay
        self._tg: asyncio.TaskGroup | None = None

    async def __aenter__(self):
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
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

            # cancellation must propagate
            except asyncio.CancelledError:
                raise

            # retryable failure
            except Exception as e:
                last_exception = e

                print(
                    f"[registry] failed to start "
                    f"extension='{extension.name}' "
                    f"attempt={attempt}/"
                    f"{self.max_start_attempts} "
                    f"error={e}"
                )

                # cleanup partial resources
                try:
                    await extension.stop()
                except Exception:
                    pass

                # retry
                if attempt < self.max_start_attempts:
                    await asyncio.sleep(self.retry_delay)

        # failed after all retries
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

        # start extension first
        await self._start_extension(extension)
        self.extensions[extension.name] = extension

        # register tools
        for local_name, tool_obj in extension.tools.items():
            full_name = f"{extension.name}.{local_name}"

            if full_name in self.tools:
                raise RuntimeError(f"Duplicate tool '{full_name}'")

            self.tools[full_name] = RegisteredTool(
                extension_name=(extension.name), tool=tool_obj
            )

    # TOOL SCHEMAS
    def get_tools(
        self, extension_name: (Optional[str]) = None
    ) -> list[LlmToolFunction]:
        tools: list[LlmToolFunction] = []
        for full_name, registered in self.tools.items():
            if extension_name and registered.extension_name != extension_name:
                continue
            tools.append(registered.tool.get_schema(full_name=full_name))
        return tools

    async def call_tool(
        self,
        fn_name: str,
        arguments: dict[str, JSONValue],
        state: Optional[HarnessState],
        llm: Optional[Llm],
        emitter: (Optional[Emitter]) = None,
    ) -> list[ObservationEvent]:
        try:
            registered = self.tools.get(fn_name)
            if registered is None:
                raise RuntimeError(f"Unknown tool '{fn_name}'")

            result = await registered.tool.run(
                arguments,
                runtime_context={"state": state, "llm": llm, "emitter": emitter},
            )
            return [ToolProduced(tool_name=fn_name, content=result)]

        except Exception as e:
            return [
                ToolFailed(
                    tool_name=fn_name, error_type=(type(e).__name__), message=str(e)
                )
            ]


# =========================================================
# EXAMPLE EXTENSION
# =========================================================


class SlackExtension(Extension):
    name = "slack"
    description = "Slack communication tools."

    async def start(self):
        self.client = object()
        print("[SlackExtension] started")

    async def stop(self):
        print("[SlackExtension] stopped")

    @tool(description="Send slack message")
    async def send(self, channel: str, message: str):
        return {
            "channel": channel,
            "message": message,
            "status": "sent",
        }


class MemoryExtension(Extension):
    name = "memory"

    @tool()
    async def remember(
        self, key: str, value: str, state: Optional[HarnessState] = None
    ):

        if state is not None:
            print(state)

        return {"stored": True}


# =========================================================
# MAIN
# =========================================================


async def main():

    async with ExtensionRegistry() as registry:
        await registry.register(SlackExtension())
        await registry.register(MemoryExtension())

        tools = registry.get_tools()

        for tool in tools:
            print(tool)

        result = await registry.call_tool(
            fn_name="slack.send",
            arguments={"channel": "#general", "message": "hello"},
            state=None,
            llm=None,
        )

        result = await registry.call_tool(
            fn_name="memory.remember",
            arguments={"key": "#general", "value": "hello"},
            state=None,
            llm=None,
        )

        print(result)


if __name__ == "__main__":
    asyncio.run(main())
