"""Deprecated compatibility shim.

Use driven.extras.extension_tool_runtime and driven.extras.utils.actor_framework instead.
"""

from driven.extras.utils.actor_framework import Actor, Registry
from driven.extras.extension_tool_runtime import (
    Extension,
    tool,
    ExtensionRegistry,
    ExtensionRequest,
    ExtensionToolRuntime,
)

__all__ = [
    "Actor",
    "Registry",
    "Extension",
    "tool",
    "ExtensionRegistry",
    "ExtensionRequest",
    "ExtensionToolRuntime",
]
