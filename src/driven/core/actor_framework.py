import logging
import functools
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, Protocol


logger = logging.getLogger("core")


# Type alias for the next function in the chain
NextCall = Callable[["Message"], Awaitable[Any]]


class Middleware(Protocol):
    async def __call__(self, msg: "Message", next_call: NextCall) -> Any: ...


@dataclass(frozen=True)
class Message:
    sender_id: str
    payload: Any
    reply_to: Optional[asyncio.Future] = None


@dataclass(frozen=True)
class Event:
    topic: str
    payload: dict


class Actor:
    def __init__(self, actor_id: str):
        self.id = actor_id
        # The mailbox now accepts Optional[Message] so we can pass None as a Poison Pill
        self.mailbox: asyncio.Queue[Optional[Message]] = asyncio.Queue()
        self._middlewares: list[Middleware] = []
        self._chain = self._build_middleware_chain()

        self._events_queues: dict[str, list[asyncio.Queue]] = {}

    def _get_queue(self, action: str):
        if action_queue := self._events_queues.get(action):
            return action_queue
        else:
            raise KeyError(f"Queue not found: {action}")

    def add_queue_to_action(self, action: str, queue: asyncio.Queue):
        if action not in self._events_queues.keys():
            self._events_queues[action] = [queue]
        else:
            self._events_queues[action].append(queue)

    def remove_queue_from_action(self, action: str, queue: asyncio.Queue):
        """Removes a subscriber queue to prevent memory leaks."""
        if action in self._events_queues and queue in self._events_queues[action]:
            self._events_queues[action].remove(queue)
            # Clean up the dictionary key if no one is listening anymore
            if not self._events_queues[action]:
                del self._events_queues[action]

    def emit(self, action: str, payload: dict):
        if not self._events_queues.get(action):
            self._events_queues[action] = []

        event = Event(topic=f"{self.id}.{action}", payload=payload)

        for queue in self._events_queues.get(action, []):
            try:
                # Try to push the event immediately without blocking
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # The subscriber is too slow. Drop the event and log it.
                logger.warning(
                    f"Dropped event '{event.topic}' for a subscriber. "
                    f"Queue is at max capacity ({queue.maxsize})."
                )

    def update_middlewares(self, middlewares: list[Middleware]):
        self._middlewares.extend(middlewares)
        self._chain = self._build_middleware_chain()

    def _build_middleware_chain(self) -> NextCall:
        # 1. The innermost core function
        async def _core_handler(msg: "Message") -> Any:
            return await self.handle_message(msg.payload)

        # Explicitly type the chain variable
        chain: NextCall = _core_handler

        # 2. Wrap from the outside in
        for mw in reversed(self._middlewares):
            # Explicitly type the closure factory and its arguments
            def _wrap(
                middleware: Middleware = mw, next_handler: NextCall = chain
            ) -> NextCall:

                @functools.wraps(
                    next_handler
                )  # Preserves function names for debuggers!
                async def _wrapped(msg: "Message") -> Any:
                    return await middleware(msg, next_handler)

                return _wrapped

            chain = _wrap()

        return chain

    async def start(self):
        """Override to initialize heavy resources."""
        pass

    async def stop(self):
        """Override to clean up resources."""
        pass

    async def handle_message(self, payload: Any) -> Any:
        raise NotImplementedError

    async def _run_loop(self):
        """The private mailbox consumer loop."""
        await self.start()
        try:
            while True:
                msg = await self.mailbox.get()

                # THE POISON PILL: If we receive None, break the loop to shut down cleanly.
                if msg is None:
                    break

                # Process concurrently so the mailbox never blocks
                asyncio.create_task(self._process_and_reply(msg))
        finally:
            await self.stop()

    async def _process_and_reply(self, msg: Message):
        """Wrapper to execute the middleware chain and fulfill the Future."""
        try:
            # We now pass the full message to the compiled chain
            result = await self._chain(msg)

            if msg.reply_to and not msg.reply_to.done():
                msg.reply_to.set_result(result)
        except Exception as e:
            if msg.reply_to and not msg.reply_to.done():
                msg.reply_to.set_exception(e)


class Registry:
    def __init__(self, default_middlewares: list[Middleware] = []):
        self.default_middlewares = default_middlewares
        self._mailboxes: dict[str, asyncio.Queue] = {}
        self._tg: Optional[asyncio.TaskGroup] = None
        self._actors: dict[str, Actor] = {}

    async def __aenter__(self):
        self._tg = asyncio.TaskGroup()
        await self._tg.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # GRACEFUL SHUTDOWN: Send a Poison Pill to every active mailbox
        for queue in self._mailboxes.values():
            await queue.put(None)

        assert self._tg
        await self._tg.__aexit__(exc_type, exc_val, exc_tb)

    async def _supervise_actor(self, actor: Actor, max_retries: int = 3):
        """The Registry handles isolated fault tolerance directly."""
        retries = 0
        while retries < max_retries:
            try:
                await actor._run_loop()
                break  # If _run_loop exits cleanly (via Poison Pill), exit the retry loop!
            except asyncio.CancelledError:
                raise
            except Exception as e:
                retries += 1
                logger.info(
                    f"[Supervisor] Actor '{actor.id}' crashed: {e}. Restarting ({retries}/{max_retries})..."
                )
                await asyncio.sleep(1)

    def register(self, actor: Actor, middlewares: list[Middleware] = []):
        if self._tg is None:
            raise RuntimeError("Registry must be used within 'async with'.")

        self._actors[actor.id] = actor
        mws = [*middlewares, *self.default_middlewares]
        actor.update_middlewares(mws)

        self._mailboxes[actor.id] = actor.mailbox
        self._tg.create_task(self._supervise_actor(actor))

    async def subscribe(self, channel: str) -> AsyncGenerator[Event, None]:
        """
        Creates a real-time event stream.
        Note: Target actors must be registered BEFORE this is called.
        """
        actor_id, action = channel.split(".")
        # Create a bounded queue that holds a maximum of 1000 pending events
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=1000)

        attached_actors: list[Actor] = []

        if actor_id == "*":
            # Attach ONLY to currently registered actors
            if not self._actors:
                logger.warning(
                    f"Subscribing to '*.{action}', but no actors are registered yet!"
                )

            for actor in self._actors.values():
                actor.add_queue_to_action(action, queue)
                attached_actors.append(actor)
        else:
            if actor := self._actors.get(actor_id):
                actor.add_queue_to_action(action, queue)
                attached_actors.append(actor)
            else:
                raise KeyError(
                    f"Cannot subscribe: Actor '{actor_id}' is not registered."
                )

        # Yield events until canceled
        try:
            while True:
                yield await queue.get()
        except asyncio.CancelledError:
            pass  # Graceful exit
        finally:
            # CLEANUP: Remove this queue from the specific actors it was attached to
            for actor in attached_actors:
                actor.remove_queue_from_action(action, queue)

    async def ask(
        self, target: str, sender_id: str, payload: Any, timeout: float = 5.0
    ) -> Any:
        if target not in self._mailboxes:
            raise ValueError(f"Target '{target}' not found.")

        reply_future = asyncio.get_running_loop().create_future()
        msg = Message(sender_id=sender_id, payload=payload, reply_to=reply_future)

        # Route to target
        await self._mailboxes[target].put(msg)

        async with asyncio.timeout(timeout):
            return await reply_future
