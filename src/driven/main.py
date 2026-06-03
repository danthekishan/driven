import logging
import asyncio
from driven.core import ExtensionRegistry, ExtensionRequest
from driven.settings import EXTENSIONS_PATH


# 1. Setup minimal logging so we can see the registry working
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


async def main():
    logger = logging.getLogger("System.Main")

    # Initialize the specific Extension Registry
    registry = ExtensionRegistry()

    async with registry:
        # ---------------------------------------------------------
        # Phase 1: Auto-Load Extensions
        # ---------------------------------------------------------
        logger.info("Scanning for extensions...")
        # This will automatically find and register 'math_expert'
        # (and any other extensions) from your extensions/ directory.
        registry.load_extensions_from_directory(EXTENSIONS_PATH)

        # ---------------------------------------------------------
        # Phase 2: Introspection (Getting Tools for an LLM)
        # ---------------------------------------------------------
        logger.info("Requesting extension info for 'math_expert'...")

        info_req = ExtensionRequest(request_type="info")

        extension_info = await registry.ask(
            target="math_expert", sender_id="main", payload=info_req
        )

        print("\n--- 🧠 Extension Introspection ---")
        print(f"Instructions: {extension_info['instructions']}")
        print(f"Available Tools: {list(extension_info['tools'].keys())}\n")

        # ---------------------------------------------------------
        # Phase 3: Executing a Tool with Context
        # ---------------------------------------------------------
        logger.info("Executing tool 'add' on 'math_expert'...")

        exec_req = ExtensionRequest(
            request_type="tool_call",
            request_body={"tool_name": "add", "kwargs": {"a": 42.5, "b": 10.5}},
            request_config={
                # Metadata the tool might need, but isn't a direct argument
                "trace_id": "req_9982",
                "user_tier": "premium",
            },
        )

        result = await registry.ask(
            target="math_expert", sender_id="main", payload=exec_req
        )

        print("--- ⚙️ Execution Result ---")
        print(f"Result: {result}\n")

        logger.info("System shutting down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
