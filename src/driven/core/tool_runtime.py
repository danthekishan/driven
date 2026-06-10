import asyncio
from typing import Callable, Optional

from driven.core.protocols import Emitter, HarnessState, Llm

from driven.core.schemas import LlmToolFunction, Message, ToolCall, ToolResult
from driven.core.tool import RegisteredTool, Tool, discover_tools


class Extension:
    name: str = "extension"
    version: str = "0.0.1"
    description: str = ""
    instructions: str = "Generic extension."
    enabled: bool = True
    requires: list[str] = []

    def __init__(self):
        self.tools: dict[str, Tool] = {t.name: t for t, _ in discover_tools(self)}
        self._create_branch: Optional[Callable] = None

    def set_create_branch(self, create_branch: Callable):
        self._create_branch = create_branch

    async def start(self):
        pass

    async def stop(self):
        pass

    async def branch(
        self,
        prompt: str | list[Message],
        system: str,
        parent_state: HarnessState,
        label: str = "",
        middlewares: list | None = None,
        session_middlewares: list | None = None,
    ) -> tuple:
        if self._create_branch is None:
            raise RuntimeError("branch not available — harness not connected")
        runner = self._create_branch(
            label=label or self.name,
            parent_state=parent_state,
            middlewares=middlewares or [],
            session_middlewares=session_middlewares or [],
        )

        return await runner(prompt=prompt, system=system)


class SubAgent(Extension):
    private_extensions: list = []

    def __init__(self):
        self._public_tools: dict[str, Tool] = {}
        self._private_tools: dict[str, tuple[Tool, str]] = {}
        self._private_ext_instances: list[Extension] = []
        self._init_tools()
        self.tools = dict(self._public_tools)

    def _init_tools(self):
        for tool_obj, is_public in discover_tools(self):
            if is_public:
                self._public_tools[tool_obj.name] = tool_obj
            else:
                self._private_tools[tool_obj.name] = (tool_obj, self.name)

    async def start(self):
        for ext in self._resolve_private_extensions():
            self._private_ext_instances.append(ext)
            await ext.start()
            for tool_obj in ext.tools.values():
                self._private_tools[tool_obj.name] = (tool_obj, ext.name)

    async def stop(self):
        for ext in self._private_ext_instances:
            try:
                await ext.stop()
            except Exception:
                pass

    def _resolve_private_extensions(self) -> list[Extension]:
        result: list[Extension] = []
        for ext in self.private_extensions:
            if isinstance(ext, Extension):
                result.append(ext)
            else:
                result.append(ext())
        return result

    async def branch(
        self,
        prompt: str | list[Message],
        system: str,
        parent_state: HarnessState,
        label: str = "",
        middlewares: list | None = None,
        session_middlewares: list | None = None,
    ) -> tuple:
        if self._create_branch is None:
            raise RuntimeError("branch not available — harness not connected")
        runner = self._create_branch(
            label=label or self.name,
            parent_state=parent_state,
            middlewares=middlewares or [],
            session_middlewares=session_middlewares or [],
            private_of=self.name,
        )
        return await runner(prompt=prompt, system=system)


class ToolRuntime:
    def __init__(
        self,
        exts: list[Extension] | None = None,
        max_start_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        self.exts = exts or []
        self.extensions: dict[str, Extension] = {}
        self.tools: dict[str, RegisteredTool] = {}
        self._private: dict[str, dict[str, RegisteredTool]] = {}
        self.max_start_attempts = max_start_attempts
        self.retry_delay = retry_delay
        self._tg: asyncio.TaskGroup | None = None

    async def connect(self, create_branch: Callable):
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
        for ext in self.exts:
            await self.register(ext, create_branch)
        return self

    async def disconnect(self, *args, **kwargs):
        for ext in self.extensions.values():
            try:
                await ext.stop()
            except Exception:
                pass
        assert self._tg
        await self._tg.__aexit__(*args, **kwargs)

    async def register(self, extension: Extension, create_branch: Callable):
        if extension.name in self.extensions:
            raise RuntimeError(f"Extension '{extension.name}' already registered")

        await self._start(extension)
        extension.set_create_branch(create_branch)
        self.extensions[extension.name] = extension
        self._register_tools(extension)

    def _register_tools(self, extension: Extension):
        for local_name, tool_obj in extension.tools.items():
            full_name = f"{extension.name}-{local_name}"
            self.tools[full_name] = RegisteredTool(extension.name, tool_obj)

        if isinstance(extension, SubAgent):
            self._private[extension.name] = {}
            for local_name, (tool_obj, source_ext) in extension._private_tools.items():
                full_name = f"{source_ext}-{local_name}"
                self._private[extension.name][full_name] = RegisteredTool(
                    source_ext, tool_obj
                )

    async def _start(self, extension: Extension):
        last_error: Exception | None = None
        for attempt in range(1, self.max_start_attempts + 1):
            try:
                await extension.start()
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                try:
                    await extension.stop()
                except Exception:
                    pass
                if attempt < self.max_start_attempts:
                    await asyncio.sleep(self.retry_delay)

        raise RuntimeError(
            f"Failed to start extension '{extension.name}' "
            f"after {self.max_start_attempts} attempts"
        ) from last_error

    def get_tools(self, private_of: Optional[str] = None) -> list[LlmToolFunction]:
        if private_of:
            pool = self._private.get(private_of, {})
            return [rt.tool.get_schema(fn) for fn, rt in pool.items()]
        return [rt.tool.get_schema(fn) for fn, rt in self.tools.items()]

    def _find(self, name: str) -> RegisteredTool | None:
        if name in self.tools:
            return self.tools[name]
        for pool in self._private.values():
            if name in pool:
                return pool[name]
        return None

    async def call_tools(
        self,
        tools: list[ToolCall],
        state: Optional[HarnessState],
        llm: Optional[Llm],
        emitter: Optional[Emitter] = None,
        timeout: Optional[float] = None,
    ) -> list[ToolResult]:
        async def _run_one(call: ToolCall) -> ToolResult:
            try:
                registered = self._find(call.name)
                if registered is None:
                    raise RuntimeError(f"Unknown tool '{call.name}'")
                result = await registered.tool.run(
                    call.arguments,
                    runtime_context={
                        "state": state,
                        "llm": llm,
                        "emitter": emitter,
                    },
                )
                return ToolResult(
                    name=call.name, ok=True, content=result, call_id=call.call_id
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
                return await asyncio.gather(*[_run_one(c) for c in tools])
        except TimeoutError:
            return [
                ToolResult(
                    name=c.name,
                    ok=False,
                    call_id=c.call_id,
                    error_type="TimeoutError",
                    error_message="tool batch timed out",
                )
                for c in tools
            ]
