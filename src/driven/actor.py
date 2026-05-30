import functools
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol


# Type alias for the next function in the chain
NextCall = Callable[["Message"], Awaitable[Any]]


class Middleware(Protocol):
    async def __call__(self, msg: "Message", next_call: NextCall) -> Any: ...


@dataclass(frozen=True)
class Message:
    sender_id: str
    payload: dict
    reply_to: Optional[asyncio.Future] = None


class Actor:
    def __init__(self, actor_id: str):
        self.id = actor_id
        # The mailbox now accepts Optional[Message] so we can pass None as a Poison Pill
        self.mailbox: asyncio.Queue[Optional[Message]] = asyncio.Queue()
        self._middlewares: list[Middleware] = []
        self._chain = self._build_middleware_chain()

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

    async def handle_message(self, payload: dict) -> Any:
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
                print(
                    f"[Supervisor] Actor '{actor.id}' crashed: {e}. Restarting ({retries}/{max_retries})..."
                )
                await asyncio.sleep(1)

    def register(self, actor: Actor, middlewares: list[Middleware] = []):
        if self._tg is None:
            raise RuntimeError("Registry must be used within 'async with'.")

        mws = [*middlewares, *self.default_middlewares]
        actor.update_middlewares(mws)

        self._mailboxes[actor.id] = actor.mailbox
        self._tg.create_task(self._supervise_actor(actor))

    async def ask(
        self, target: str, sender_id: str, payload: dict, timeout: float = 5.0
    ) -> Any:
        if target not in self._mailboxes:
            raise ValueError(f"Target '{target}' not found.")

        reply_future = asyncio.get_running_loop().create_future()
        msg = Message(sender_id=sender_id, payload=payload, reply_to=reply_future)

        # Route to target
        await self._mailboxes[target].put(msg)

        async with asyncio.timeout(timeout):
            return await reply_future
