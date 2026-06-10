# driven

a protocol-driven agent orchestration framework for building LLM-powered tool-using agents.

## what it is

driven is a framework — not an agent, not a runtime, not a set of tools.

it provides the orchestration layer: protocols, schemas, a composable execution model, and a wiring mechanism. you bring the LLM, the tools, the state strategy, and the control logic. driven runs them.

## what driven provides

| component                 | what it is                                                               |
| ------------------------- | ------------------------------------------------------------------------ |
| `Agent`                   | wiring layer — composes harness, extensions, controller, state, middlewares |
| `Harness`                 | execution loop — runs turns, applies middlewares, manages lifecycle, owns ToolRuntime |
| `ToolRuntime`             | manages extension lifecycle, registration, tool discovery, and execution |
| `Controller` (protocol)   | decides what happens next in a turn                                      |
| `StateManager` (protocol) | persists and recovers state                                              |
| `Llm` (protocol)          | talks to a language model                                                |
| `Emitter` (protocol)      | observes events                                                          |
| `Extension` / `@tool()`   | base class and decorator for building tool groups                        |
| `SubAgent`                | extension that spawns branches with private tools                        |
| middlewares               | intercept hooks for compaction, limits, retries, logging, etc.           |

Using `Extension` base class and `@tool()` decorator to define custom tools:

```python
from driven.core.tool_runtime import Extension
from driven.core.tool import tool
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

Wire everything together with `Agent`:

```python
from driven.agent import Agent, ToolCallingController, InMemoryStateManager
from driven.core.harness_middlewares import max_steps_middleware, compaction_step_middleware

async with Agent(
    llm=MyLlm(...),
    extensions=[SearchExtension()],
    controller=ToolCallingController(),
    state_manager=InMemoryStateManager(),
    middlewares=[max_steps_middleware(25), compaction_step_middleware()],
) as agent:
    state, error = await agent.run(prompt="Search for ...")
```

## branches and sub-agents

the harness supports **branching** — spawning a separate execution run from within a tool call. branches get their own state and middleware chain, but share the parent's services (LLM, emitter).

`SubAgent` is an `Extension` that uses branching to spawn isolated agent runs with **private tools** — tools the parent agent never sees:

```python
from driven.core.tool_runtime import SubAgent
from driven.core.tool import tool
from driven.core.protocols import HarnessState

class CodingAgent(SubAgent):
    name = "coding-agent"
    description = "Spawns a coding agent to perform tasks."
    private_extensions = [CoderExtension(workspace="./workspace")]

    @tool(description="Run a coding task", public=True)
    async def run(self, input: CodeTaskInput, state: HarnessState) -> dict:
        branch_state, error = await self.branch(
            prompt=input.task,
            system="You are a coding agent. Perform the task using available tools.",
            parent_state=state,
        )
        ...
```

- **public tools** (`@tool(public=True)`) — visible to the parent agent, appear as the sub-agent's interface
- **private tools** (`@tool()`) — only visible inside the branch, come from `private_extensions`
- the parent agent sees only the sub-agent's public tools — it delegates, the branch executes
- extensions call `self.branch()` directly — no injection needed

branch events carry metadata (`branch_id`, `parent_run_id`, `parent_step`, `label`) so you can trace the hierarchy.

## architecture

```
                    ┌─────────────┐
                    │    Agent    │  wiring — you compose this
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │   Harness   │  execution loop — driven owns this
                    └──────┬──────┘
                           │
              ┌────────────┼───────────┐
              │            │           │
        ┌─────┴─────┐ ┌────┴───┐ ┌─────┴──────┐
        │ Controller │ │  Llm  │ │ ToolRuntime │
        └───────────┘ └────────┘ └─────┬──────┘
                                       │
                              ┌────────┴────────┐
                              │   Extensions    │  you own these
                              └────────┬────────┘
                                       │
                              ┌────────┴──────────┐
                              │  SubAgent (branch)│  private tools, own state
                              └───────────────────┘
```

### harness

the execution engine. it runs a loop: ask the controller for a turn, process events, update state, repeat until done.

middleware chains wrap the loop — step-level and session-level — giving you hooks for compaction, limits, logging, retries, or anything else without touching core logic.

### tool runtime

the boundary between the harness and the outside world. discovers available tools and executes them.

`ToolRuntime` is the provided implementation:

- **extensions** are scoped, named groups of tools — each with a lifecycle (`start`/`stop`) and auto-discovered `@tool()`-decorated methods
- the runtime manages extension lifecycle, tool registration, and parallel execution
- tools can receive runtime context (`state`, `llm`, `emitter`) via parameter injection — no coupling
- extensions get branching capability via `self.branch()` — wired by the harness, no injection needed

### controller

decides what happens each turn. streams `TurnEvent`s — request prepared, output received, tools requested, tools completed, assistant finalized, turn finished.

`ToolCallingController` is the provided implementation: a standard LLM → tool call → tool result → repeat loop.

the protocol allows anything — reAct loops, planners, multi-agent delegation, reflective retries.

### state manager

state persistence is a protocol. `InMemoryStateManager` and `LocalStateManager` (file-based) are provided. the harness loads and saves state automatically — enabling crash recovery, replay, and resumable runs through the protocol, not through built-in complexity.

### middlewares

middlewares wrap the execution loop, giving you intercept points without modifying core behavior:

- **step middlewares** — wrap each individual step (e.g., compaction, max steps)
- **session middlewares** — wrap the entire run loop (e.g., lifecycle, logging)

they compose. they're explicit. there are no hidden defaults.

## examples

```bash
# number guessing game — agent uses extension tools directly
uv run python examples/main.py --name number-guess

# coding agent — SubAgent with private CoderExtension tools
uv run python examples/main.py --name coding-agent
```

| example        | what it shows                                                                  |
| -------------- | ------------------------------------------------------------------------------ |
| `number-guess` | agent plays a game using `NumberGuessExtension` tools directly                 |
| `coding-agent` | `CodingAgent` SubAgent — parent sees 1 tool, branch gets 7 private coder tools |

## design principles

- **protocol over implementation** — contracts first, fill in later
- **explicit over implicit** — no hidden defaults, no magic wiring
- **extendability over convenience** — the framework provides hooks, not opinions
- **decoupling over integration** — harness doesn't know about tools; runtime doesn't know about orchestration
- **minimal until proven otherwise** — add complexity when real needs emerge, not in anticipation

## status

early exploration. the architecture is solid, the abstractions are validated, but the surface is still forming.

current focus:

- building real agents to validate the abstractions
- discovering practical limitations
- refining the protocol boundaries
- keeping it minimal until the picture is clear
