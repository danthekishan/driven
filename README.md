# driven

a protocol-driven agent orchestration framework for building LLM-powered tool-using agents.

## what it is

driven is a framework — not an agent, not a runtime, not a set of tools.

it provides the orchestration layer: protocols, schemas, a composable execution model, and a wiring mechanism. you bring the LLM, the tools, the state strategy, and the control logic. driven runs them.

## what driven provides

| component | what it is |
|---|---|
| `Agent` | wiring layer — composes harness, runtime, controller, state, middlewares |
| `Harness` | execution loop — runs turns, applies middlewares, manages lifecycle |
| `Controller` (protocol) | decides what happens next in a turn |
| `Runtime` (protocol) | discovers and executes tools |
| `StateManager` (protocol) | persists and recovers state |
| `Llm` (protocol) | talks to a language model |
| `Emitter` (protocol) | observes events |
| `TraceSink` (protocol) | records execution traces |
| `Extension` / `@tool` | base class and decorator for building tool groups |
| `ExtensionRegistry` | manages extension lifecycle, registration, and parallel execution |
| middlewares | intercept hooks for compaction, limits, retries, logging, etc. |

## what you provide

driven ships no tools, no extensions, no LLM clients. those belong to your application.

the framework gives you the `Extension` base class and `@tool` decorator to define your own:

```python
from driven.core.tool_runtime import Extension, tool
from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str

class SearchExtension(Extension):
    name = "search"
    description = "Search tools."

    @tool(description="Search the web.")
    async def search_web(self, input: SearchInput) -> str:
        ...  # your logic
```

you wire everything together with `Agent`:

```python
from driven.agent import Agent, ToolCallingController, InMemoryStateManager
from driven.core.harness_middlewares import max_steps_middleware, compaction_step_middleware
from driven.core.tool_runtime import ExtensionRegistry

async with Agent(
    llm=MyLlm(...),
    runtime=ExtensionRegistry(exts=[SearchExtension()]),
    controller=ToolCallingController(),
    state_manager=InMemoryStateManager(),
    middlewares=[max_steps_middleware(25), compaction_step_middleware()],
) as agent:
    state, error = await agent.run(prompt="Search for ...")
```

everything explicit. nothing hidden. you see every component that's wired in.

## architecture

```
                    ┌─────────────┐
                    │    Agent     │  wiring — you compose this
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Harness    │  execution loop — driven owns this
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │ Controller │ │  Llm  │ │  Runtime  │
        └───────────┘ └───────┘ └─────┬─────┘
                                       │
                              ┌────────┴────────┐
                              │ ExtensionRegistry │
                              └────────┬────────┘
                                       │
                              ┌────────┴────────┐
                              │   Extensions     │  you own these
                              └─────────────────┘
```

### harness

the execution engine. it runs a loop: ask the controller for a turn, process events, update state, repeat until done.

middleware chains wrap the loop — step-level and session-level — giving you hooks for compaction, limits, logging, retries, or anything else without touching core logic.

### controller

decides what happens each turn. streams `TurnEvent`s — request prepared, output received, tools requested, tools completed, assistant finalized, turn finished.

`ToolCallingController` is the provided implementation: a standard LLM → tool call → tool result → repeat loop.

the protocol allows anything — reAct loops, planners, multi-agent delegation, reflective retries.

### runtime

the boundary between the harness and the outside world. discovers available tools and executes them.

`ExtensionRegistry` is the provided implementation:

- **extensions** are scoped, named groups of tools — each with a lifecycle (`start`/`stop`) and auto-discovered `@tool`-decorated methods
- the registry manages extension lifecycle, tool registration, and parallel execution
- tools can receive runtime context (`state`, `llm`, `emitter`) via parameter injection — no coupling
- can be extended to share resources (clients, connection pools) across extensions

a runtime can be local (extensions in-process), remote, or anything else. the harness doesn't know or care.

### state manager

state persistence is a protocol. `InMemoryStateManager` and `LocalStateManager` (file-based) are provided. the harness loads and saves state automatically — enabling crash recovery, replay, and resumable runs through the protocol, not through built-in complexity.

### middlewares

middlewares wrap the execution loop, giving you intercept points without modifying core behavior:

- **step middlewares** — wrap each individual step (e.g., compaction, max steps)
- **session middlewares** — wrap the entire run loop (e.g., lifecycle, logging)

they compose. they're explicit. there are no hidden defaults.

## design principles

- **protocol over implementation** — contracts first, fill in later
- **explicit over implicit** — no hidden defaults, no magic wiring
- **extendability over convenience** — the framework provides hooks, not opinions
- **decoupling over integration** — harness doesn't know about tools; runtime doesn't know about orchestration
- **minimal until proven otherwise** — add complexity when real needs emerge, not in anticipation

## examples

- [`examples/minimal_coder/`](examples/minimal_coder/) — minimal agent using the coder extension with OpenAI

## status

early exploration. the architecture is solid, the abstractions are validated, but the surface is still forming.

current focus:

- building real agents to validate the abstractions
- discovering practical limitations
- refining the protocol boundaries
- keeping it minimal until the picture is clear
