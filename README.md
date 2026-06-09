# driven

A protocol-driven agent orchestration framework for building LLM-powered tool-using agents.

## what it is

driven is not an agent. it is the infrastructure for building one.

it provides a small set of protocols, schemas, and a composable execution model that lets you wire together LLMs, tools, state management, and control strategies — without the framework making assumptions about what you're building.

## core idea

**protocols define contracts. implementations fill them. the framework orchestrates.**

every major component in driven is defined as a protocol:

- **`Llm`** — how to talk to a language model
- **`Controller`** — how to decide what happens next in a turn
- **`Runtime`** — how to discover and execute tools
- **`StateManager`** — how to persist and recover state
- **`Emitter`** — how to observe what's happening
- **`TraceSink`** — how to record execution traces

the framework doesn't care how you implement these. it only cares that they satisfy the contract.

## architecture

```
                    ┌─────────────┐
                    │    Agent     │  wiring layer — composes everything
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Harness    │  execution loop — runs turns via controller
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
                              │   Extensions     │  logical groups of tools
                              └─────────────────┘
```

### harness

the harness is the execution engine. it runs a loop: ask the controller for a turn, process events, update state, repeat until done.

middleware chains wrap the loop — step-level and session-level — giving you hooks for compaction, limits, logging, retries, or anything else without touching core logic.

### controller

the controller decides what happens each turn. it streams `TurnEvent`s — request prepared, output received, tools requested, tools completed, assistant finalized, turn finished.

`ToolCallingController` is the provided implementation: a standard LLM → tool call → tool result → repeat loop.

but the protocol allows anything — reAct loops, planners, multi-agent delegation, reflective retries.

### runtime (tool runtime)

the runtime is the boundary between the harness and the outside world. it discovers available tools and executes them.

`ExtensionRegistry` is the provided implementation:

- **extensions** are logical groups of tools — each with a name, lifecycle (`start`/`stop`), and auto-discovered `@tool`-decorated methods
- the registry manages extension lifecycle, tool registration, and parallel execution
- tools can receive runtime context (`state`, `llm`, `emitter`) via parameter injection — no coupling

this separation is intentional:

- **harness logic** belongs to driven — reliability, extendability, correctness
- **tool/runtime logic** belongs to the user — what tools exist, how they work, what resources they need

a runtime can be local (extensions in-process), remote, or anything else. the harness doesn't know or care.

### extensions

an extension is a scoped, named group of tools. think of it as a plugin:

```
coder          → read_file, write_file, run_command, ...
github         → create_pr, list_issues, ...
database       → query, insert, ...
```

each extension owns its scope. the registry can be extended to share common resources (clients, connection pools) across extensions — avoiding duplicate connections and boilerplate.

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

## status

this is early exploration. the architecture is solid, the abstractions are validated, but the surface is still forming.

the current focus:

- building real agents to validate the abstractions
- discovering practical limitations
- refining the protocol boundaries
- keeping it minimal until the picture is clear
