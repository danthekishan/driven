import sys
import inspect
import logging
import asyncio
import importlib.util
from pathlib import Path
from typing import Callable, Optional, Any, Literal
from dataclasses import dataclass, field

from driven.core.actor_framework import Actor, Registry

logger = logging.getLogger("core")


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to mark an Extension method as an executable tool."""

    def decorator(func):
        func.__is_tool__ = True
        func.__tool_name__ = name or func.__name__
        func.__tool_description__ = (
            description or func.__doc__ or "No description provided."
        )
        return func

    return decorator


@dataclass(frozen=True)
class ExtensionRequest:
    # 'info' for introspection, 'tool_call' for execution
    request_type: Literal["info", "tool_call"]

    # The actual arguments (e.g., {"tool_name": "add", "kwargs": {"a": 1, "b": 2}})
    request_body: dict[str, Any] = field(default_factory=dict)

    # Metadata: auth tokens, tracing IDs, LLM context, or specific tool timeouts
    request_config: dict[str, Any] = field(default_factory=dict)


class Extension(Actor):
    instructions: str = "You are a generic extension."
    default_id: str = "generic_extension"

    def __init__(self, actor_id: Optional[str] = None):
        super().__init__(actor_id or self.default_id)
        self.tools: dict[str, Callable] = {}
        self._discover_tools()

    def _discover_tools(self):
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if getattr(method, "__is_tool__", False):
                self.tools[method.__tool_name__] = method  # type: ignore

    async def handle_message(self, payload: ExtensionRequest | dict) -> Any:
        """The standardized message contract for Extensions."""

        # 1. Gracefully handle raw dicts by casting them to our strict contract
        if isinstance(payload, dict):
            payload = ExtensionRequest(**payload)

        # 2. Handle Introspection (get_info)
        if payload.request_type == "info":
            return {
                "instructions": self.instructions,
                "tools": {
                    name: func.__tool_description__  # type: ignore
                    for name, func in self.tools.items()
                },
            }

        # 3. Handle Execution (tool_call)
        elif payload.request_type == "tool_call":
            tool_name = payload.request_body.get("tool_name")
            kwargs = payload.request_body.get("kwargs", {})

            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' not found in extension '{self.id}'")

            tool_func = self.tools[tool_name]

            # Pass the `request_config` down if the tool needs it
            if payload.request_config.get("inject_config", False):
                kwargs["_config"] = payload.request_config

            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                return await tool_func(**kwargs)
            else:
                return tool_func(**kwargs)

        else:
            raise ValueError(f"Unknown request_type: {payload.request_type}")


class ExtensionRegistry(Registry):
    def load_extensions_from_directory(self, directory_path: str | Path):
        """Dynamically loads and registers all Extension classes from a directory."""
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Extensions directory '{dir_path}' not found.")

        # Iterate through all .py files in the directory
        for file_path in dir_path.glob("*.py"):
            if file_path.name == "__init__.py":
                continue

            # 1. Dynamically load the module
            module_name = f"extensions_module_{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            assert spec

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            assert spec.loader
            spec.loader.exec_module(module)

            # 2. Inspect the module for Extension subclasses
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Ensure it's a subclass of Extension, but NOT the base Extension class itself
                if issubclass(obj, Extension) and obj is not Extension:
                    # Instantiate the extension (uses its default_id)
                    extension_instance = obj()

                    # Register it using the parent Registry's logic
                    self.register(extension_instance)
                    logger.info(
                        f"[ExtensionRegistry] Auto-loaded extension: '{extension_instance.id}' (Tools: {list(extension_instance.tools.keys())})"
                    )
