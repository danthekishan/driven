# Driven Harness — Future Improvements & Architectural Roadmap

## Purpose

This document captures architectural improvements and future directions for the current Harness-based agent runtime system.

These are intentionally **not immediate implementation tasks**.

The current focus should remain:

- building real agents
- validating abstractions
- testing execution behavior
- discovering practical limitations
- stress testing orchestration flows

This document exists to preserve architectural ideas and future enhancements discovered during early experimentation.

---

# Current Strengths of the System

The current architecture already demonstrates strong separation between:

- orchestration
- decision making
- runtime execution
- tool management
- state persistence
- middleware
- observability

The core execution loop is intentionally minimal and composable.

Key strengths already validated:

- protocol-driven design
- middleware execution model
- extension lifecycle management
- runtime context injection
- action/observation separation
- persistent state abstraction
- event emission architecture
- deterministic testing via scripted LLMs

The system is already capable of supporting:

- simple agents
- multi-step tool execution
- durable workflows
- plugin ecosystems
- distributed runtimes
- streaming/event-driven interfaces

---

# Architectural Improvements

---

# 1. Multi-Tool Action Support

## Current State

`LlmOutput` currently supports only a single tool call:

```python
tool_call: Optional[ToolCall]
```

This limits execution to one tool action per decision cycle.

---

## Future Improvement

Replace:

```python
tool_call: Optional[ToolCall]
```

with:

```python
tool_calls: list[ToolCall]
```

This enables:

- multiple tool calls per step
- parallel tool execution
- batched planning
- more advanced agent strategies

---

## Benefits

### Parallel Execution

Independent tools can run concurrently:

```text
search_docs()
search_web()
search_db()
```

instead of sequentially.

---

### Reduced LLM Round Trips

The controller can request multiple operations at once.

---

### Better Planner Integration

More advanced planners often generate execution batches.

---

## Future Possibilities

- dependency graphs
- DAG execution
- speculative execution
- parallel orchestration
- map/reduce style workflows

---

# 2. Explicit Completion & Stop Reasons

## Current State

Execution completion is inferred implicitly.

Example:

```python
AssistantResp → state.done = True
```

---

## Future Improvement

Introduce explicit termination metadata.

Example:

```python
state.done_reason
```

Possible values:

- completed
- cancelled
- timeout
- max_steps
- error
- interrupted
- budget_exceeded
- user_stop

---

## Benefits

### Better Observability

Understanding _why_ a run ended becomes easier.

---

### Improved Reliability

Prevents ambiguous termination states.

---

### Workflow Integration

External systems often need explicit completion states.

---

# 3. Max Step Protection

## Current State

Execution loop:

```python
while not state.done:
```

can theoretically run forever.

---

## Future Improvement

Introduce configurable execution limits.

Example:

```python
max_steps=25
```

Possible implementation locations:

- middleware
- session policy
- harness config

---

## Benefits

- prevents infinite loops
- safer production execution
- protects against bad prompts/planners
- improves resource management

---

# 4. Runtime / Registry Separation

## Current State

`ExtensionRegistry` currently handles:

- extension lifecycle
- tool registration
- tool execution
- runtime coordination

This is acceptable for current scale.

---

## Future Improvement

Split responsibilities into dedicated components.

Possible structure:

```text
ExtensionLifecycleManager
ToolRegistry
RuntimeExecutor
ToolResolver
```

---

## Benefits

### Better Scalability

Supports distributed runtimes more cleanly.

---

### Cleaner Architecture

Single responsibility becomes clearer.

---

### Easier Testing

Each component can be validated independently.

---

# 5. Structured Event System

## Current State

Events are emitted using string names:

```python
emit("actions", ...)
emit("observations", ...)
```

---

## Future Improvement

Introduce typed event models.

Example:

```python
ToolStartedEvent
ToolCompletedEvent
StepAdvancedEvent
RunStartedEvent
```

---

## Benefits

### Stronger Contracts

Improves reliability and discoverability.

---

### Better Streaming

Structured events work better with:

- WebSockets
- SSE
- dashboards
- tracing systems

---

### Easier Integrations

External systems can consume strongly typed events.

---

# 6. Parallel Tool Execution

## Current State

Actions execute sequentially:

```python
for action in actions:
```

---

## Future Improvement

Introduce concurrent execution for independent actions.

Possible implementation:

```python
asyncio.gather(...)
```

---

## Important Considerations

Parallel execution introduces:

- ordering concerns
- shared state access
- cancellation propagation
- timeout coordination
- dependency management

This should be introduced carefully.

---

## Benefits

- faster execution
- improved throughput
- lower latency
- scalable orchestration

---

# 7. Streaming Execution Model

## Current State

The framework supports streaming LLM interfaces but the Harness loop itself is batch-oriented.

---

## Future Improvement

Allow streaming observations/events during execution.

Possible features:

- token streaming
- incremental tool output
- partial observations
- live UI updates
- progressive execution rendering

---

## Benefits

### Better User Experience

Improved responsiveness for long-running agents.

---

### Real-Time Interfaces

Supports:

- terminal UIs
- web dashboards
- chat interfaces
- live agent inspection

---

# 8. Distributed Runtime Execution

## Current State

Tool execution is local to the current process.

---

## Future Improvement

Allow runtimes to delegate execution remotely.

Possible transports:

- NATS
- gRPC
- HTTP
- MCP
- queues
- actor systems

---

## Benefits

### Horizontal Scaling

Run tools across multiple workers/machines.

---

### Isolation

Heavy tools can run independently.

---

### Better Resource Management

GPU tools, browser tools, etc. can run separately.

---

# 9. Persistent Workflow Execution

## Current State

The architecture already supports persistence through `StateManager`.

---

## Future Improvement

Expand persistence into durable workflow execution.

Possible features:

- resumable runs
- checkpointing
- replay
- crash recovery
- execution history
- workflow snapshots

---

## Benefits

### Reliability

Long-running agents survive crashes/restarts.

---

### Auditability

Complete replay/debugging support.

---

### Production Readiness

Necessary for serious orchestration systems.

---

# 10. Middleware Ecosystem Expansion

## Current State

Middleware system is already flexible and powerful.

---

## Future Improvements

Potential middleware types:

### Reliability

- retries
- timeout policies
- circuit breakers

### Observability

- tracing
- metrics
- structured logging

### Safety

- guardrails
- policy enforcement
- tool restrictions

### Cost Control

- token budgets
- cost tracking
- execution limits

### Memory Management

- summarization
- vector memory
- adaptive compaction

### Scheduling

- rate limiting
- concurrency limits
- prioritization

---

# 11. Tool Dependency & Planning System

## Current State

Actions are currently independent.

---

## Future Improvement

Allow actions to express dependencies.

Example:

```text
search_docs
    ↓
extract_entities
    ↓
query_database
```

This would naturally evolve into:

- execution graphs
- workflow DAGs
- planner outputs
- orchestrated pipelines

---

# 12. Tool Capability Metadata

## Current State

Tools expose:

- name
- description
- schema

---

## Future Improvement

Expand metadata with capability information.

Examples:

```python
supports_streaming=True
cost_estimate=...
requires_network=True
side_effects=True
```

---

## Benefits

### Smarter Planning

Controllers can reason about:

- expensive tools
- unsafe tools
- slow tools
- streaming tools

---

### Better Runtime Policies

Runtime can enforce capability restrictions.

---

# 13. Typed State & State Reducers

## Current State

State updates occur directly inside the Harness loop.

---

## Future Improvement

Introduce reducer-style state transitions.

Example:

```python
reduce(state, observation)
```

---

## Benefits

### Better Determinism

Centralized state transition logic.

---

### Easier Replay

Reducer systems replay naturally.

---

### Cleaner Event Sourcing

Fits action/observation architecture well.

---

# 14. Advanced Controller Types

## Current State

Current controller is simple LLM → action mapping.

---

## Future Possibilities

### ReAct Controllers

Thought/action loops.

---

### Planner Controllers

Multi-step planning.

---

### Graph Controllers

Execution graph generation.

---

### Multi-Agent Controllers

Delegation/routing between agents.

---

### Reflective Controllers

Self-evaluation and retries.

---

# 15. Improved Cancellation & Timeouts

## Current State

Cancellation handling already exists partially.

---

## Future Improvement

Introduce structured cancellation propagation.

Features:

- parent/child cancellation
- cascading timeouts
- cooperative tool cancellation
- interruptible execution

---

# Recommended Near-Term Priority

At the current stage, the focus should remain:

## Priority

1. build real agents
2. validate execution behavior
3. stress test orchestration
4. identify practical bottlenecks
5. discover abstraction weaknesses
6. experiment with runtime patterns

---

## Avoid Premature Complexity

The current architecture is intentionally lightweight and clean.

Many improvements above should only be implemented once:

- real limitations appear
- scaling needs emerge
- orchestration complexity becomes concrete

---

# Final Perspective

The current system already provides a strong foundation for:

- agent runtimes
- orchestration frameworks
- workflow engines
- tool ecosystems
- distributed execution systems

The architecture demonstrates strong separation of concerns and composability.

Future improvements should preserve these qualities while expanding execution capabilities incrementally.
