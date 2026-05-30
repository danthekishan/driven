import asyncio
from typing import Any
import pytest
from driven.actor import Actor, Registry

pytestmark = pytest.mark.asyncio


class EchoActor(Actor):
    async def handle_message(self, payload: dict) -> Any:
        return f"echo: {payload['text']}"


class InfrastructureCrashActor(Actor):
    """Simulates an actor whose core infrastructure breaks, triggering a restart."""

    def __init__(self, actor_id: str):
        super().__init__(actor_id)
        self.boot_count = 0

    async def _run_loop(self):
        self.boot_count += 1
        if self.boot_count == 1:
            # Crash the mailbox loop on the first boot
            raise RuntimeError("Database connection dropped!")

        # On the second boot (after supervisor restarts it), run normally
        await super()._run_loop()

    async def handle_message(self, payload: dict) -> Any:
        return "Recovered!"


class DelayActor(Actor):
    async def handle_message(self, payload: dict) -> float:
        delay = payload["delay"]
        await asyncio.sleep(delay)
        return delay


async def test_successful_round_trip_communication():
    async with Registry() as registry:
        registry.register(EchoActor("actor-echo"))

        result = await registry.ask("actor-echo", "test", {"text": "hello world"})
        assert result == "echo: hello world"


async def test_supervisor_actor_resurrection():
    async with Registry() as registry:
        flaky = InfrastructureCrashActor("actor-crash")
        registry.register(flaky)

        # Give the supervisor 1 second to catch the crash and reboot the actor
        await asyncio.sleep(1)

        # The actor should now be on its second life and accepting messages
        result = await registry.ask("actor-crash", "test", {"text": "hello"})

        assert result == "Recovered!"
        assert flaky.boot_count == 2


async def test_mailbox_non_blocking_concurrency():
    async with Registry() as registry:
        registry.register(DelayActor("actor-delay"))

        # Fire a slow request and a fast request simultaneously
        slow_task = asyncio.create_task(
            registry.ask("actor-delay", "test", {"delay": 1.0})
        )
        fast_task = asyncio.create_task(
            registry.ask("actor-delay", "test", {"delay": 0.1})
        )

        done, pending = await asyncio.wait(
            [slow_task, fast_task], return_when=asyncio.FIRST_COMPLETED
        )

        # The fast task should ALWAYS win if concurrency is working
        assert fast_task in done
        assert slow_task in pending

        await slow_task  # Clean up
