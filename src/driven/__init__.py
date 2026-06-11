import logging
import sys

from driven.agent.agent import Agent
from driven.agent.controller import ToolCallingController
from driven.agent.state_manager import InMemoryStateManager, LocalStateManager
from driven.core.schemas import RunOpts
from driven.core.protocols import Emitter, HarnessState
from driven.core.tool_runtime import Extension, SubAgent
from driven.core.tool import tool
from driven.core.harness_middlewares import (
    compaction_step_middleware,
    max_steps_middleware,
)


def setup_logging():
    """Configure this once at the startup of your application."""
    logging.basicConfig(
        level=logging.INFO,  # Minimum level to display (DEBUG, INFO, WARNING, ERROR)
        format="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)  # Outputs to console
            # logging.FileHandler("actor_system.log") # Uncomment to also write to a file
        ],
    )


setup_logging()

__all__ = [
    "HarnessState",
    "Agent",
    "RunOpts",
    "ToolCallingController",
    "InMemoryStateManager",
    "LocalStateManager",
    "Emitter",
    "compaction_step_middleware",
    "max_steps_middleware",
    "tool",
    "Extension",
    "SubAgent",
]
