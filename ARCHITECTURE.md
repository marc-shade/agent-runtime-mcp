# Agent Runtime MCP - Architecture Documentation

## Overview

Agent Runtime MCP provides persistent task management and multi-agent orchestration for autonomous AI systems. It enables Claude Code and other AI agents to maintain continuity across sessions through persistent goal/task storage and structured agent pipelines.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Claude Code / AI Client                       │
│                    (MCP Client - Initiates Tool Calls)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ MCP Protocol (stdio/SSE)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Agent Runtime MCP Server                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │ Goal Manager  │  │ Task Scheduler│  │ Relay Race Protocol   │   │
│  │               │  │               │  │                       │   │
│  │ - create_goal │  │ - get_next    │  │ - create_pipeline     │   │
│  │ - decompose   │  │ - update      │  │ - advance_relay       │   │
│  │ - list_goals  │  │ - list_tasks  │  │ - get_baton           │   │
│  └───────┬───────┘  └───────┬───────┘  └───────────┬───────────┘   │
│          │                  │                      │                │
│          │    ┌─────────────┴──────────────┐      │                │
│          │    │      Circuit Breaker       │      │                │
│          │    │   (Fault Tolerance Layer)  │      │                │
│          │    │                            │      │                │
│          │    │ - status/list/trip/reset   │      │                │
│          │    │ - record_failure/success   │      │                │
│          │    │ - configure thresholds     │      │                │
│          │    └─────────────┬──────────────┘      │                │
│          │                  │                      │                │
│          └──────────────────┼──────────────────────┘                │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         SQLite Database                              │
│                    (~/.claude/agent_runtime.db)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   goals     │  │   tasks     │  │ task_queue  │  │  relays    │ │
│  │             │  │             │  │             │  │            │ │
│  │ id          │  │ id          │  │ id          │  │ id         │ │
│  │ name        │  │ goal_id     │  │ task_id     │  │ name       │ │
│  │ description │  │ title       │  │ position    │  │ status     │ │
│  │ status      │  │ status      │  │ scheduled   │  │ steps      │ │
│  │ metadata    │  │ priority    │  │             │  │ baton      │ │
│  └─────────────┘  │ dependencies│  └─────────────┘  └────────────┘ │
│                   └─────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Goal Manager

Manages high-level objectives that persist across sessions.

```python
# Data Model
Goal:
  id: int (auto)
  name: str
  description: str
  status: 'active' | 'completed' | 'cancelled'
  created_at: timestamp
  completed_at: timestamp (nullable)
  metadata: JSON (tags, context, etc.)
```

**Key Operations:**
- `create_goal`: Define new objective with name and description
- `decompose_goal`: AI-powered breakdown into actionable tasks
- `list_goals`: Query goals by status
- `get_goal`: Retrieve goal details

### 2. Task Scheduler

Priority-based task management with dependency tracking.

```python
# Data Model
Task:
  id: int (auto)
  goal_id: int (FK to goals)
  title: str
  description: str
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  priority: int (1-10, 10 = highest)
  dependencies: JSON array of task IDs
  result: str (completion output)
  error: str (failure reason)
```

**Key Operations:**
- `create_task`: Manually add task with priority and dependencies
- `get_next_task`: Get highest priority task with dependencies met
- `update_task_status`: Transition task state
- `list_tasks`: Query tasks by goal or status

**Task State Machine:**
```
                 ┌──────────────┐
                 │   pending    │
                 └──────┬───────┘
                        │ get_next_task
                        ▼
                 ┌──────────────┐
                 │ in_progress  │
                 └──────┬───────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
           ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │completed │ │  failed  │ │cancelled │
    └──────────┘ └──────────┘ └──────────┘
```

### 3. Relay Race Protocol

Multi-agent pipeline orchestration with structured handoffs.

```python
# Data Model
RelayPipeline:
  id: str (UUID)
  name: str
  goal: str
  status: 'pending' | 'in_progress' | 'completed' | 'failed'
  current_step: int
  total_steps: int
  agent_types: List[str]  # ["researcher", "analyzer", ...]
  token_budget: int
  tokens_used: int

RelayBaton:
  pipeline_id: str
  step_index: int
  agent_type: str
  input_summary: str
  output_entity_id: int (memory reference)
  quality_score: float (0-1)
  l_score: float (0-1, provenance)
```

**Agent Types:**
| Type | Role | Typical Output |
|------|------|----------------|
| researcher | Gather information | Raw data, sources |
| analyzer | Process and analyze | Insights, patterns |
| synthesizer | Combine findings | Unified view |
| validator | Verify quality | Corrections, scores |
| formatter | Polish output | Final format |
| domain_expert | Domain-specific | Specialized knowledge |

**Baton Handoff Flow:**
```
Step N Agent                     Step N+1 Agent
     │                                │
     │ Complete work                  │
     │                                │
     ▼                                │
advance_relay(                        │
  quality_score=0.85,                 │
  l_score=0.9,                        │
  output_entity_id=123,               │
  tokens_used=5000,                   │
  output_summary="..."                │
)                                     │
     │                                │
     │          Baton passed          │
     └───────────────────────────────►│
                                      │
                              get_relay_baton()
                                      │
                                      ▼
                              Process with context
```

### 4. Circuit Breaker

Fault tolerance pattern preventing cascading failures.

```python
# Data Model
CircuitBreaker:
  agent_id: str
  state: 'CLOSED' | 'OPEN' | 'HALF_OPEN'
  failure_count: int
  last_failure: timestamp
  failure_threshold: int (default: 5)
  window_seconds: int (default: 60)
  cooldown_seconds: int (default: 300)
  fallback_agent: str
```

**State Transitions:**
```
       CLOSED (Normal)
            │
            │ failure_count >= threshold
            ▼
         OPEN (Blocking)
            │
            │ cooldown_seconds elapsed
            ▼
      HALF_OPEN (Testing)
           / \
          /   \
    success   failure
        /       \
       ▼         ▼
    CLOSED     OPEN
```

**Failure Types:**
- `timeout`: Agent took too long
- `exception`: Agent threw error
- `quality_failure`: Output below threshold
- `resource_exhausted`: Token budget exceeded
- `rate_limited`: API rate limit hit
- `invalid_output`: Output format invalid

## Integration Patterns

### With Enhanced Memory MCP

Store and retrieve agent outputs in persistent memory:

```python
# After agent produces output
entity = enhanced_memory.create_entities([{
    "name": f"relay-output-{pipeline_id}-step-{step}",
    "entityType": "relay_output",
    "observations": [output_summary, raw_output]
}])

# Pass to next agent
advance_relay(
    output_entity_id=entity["id"],
    ...
)

# Next agent retrieves context
context = enhanced_memory.search_nodes(
    query=f"relay_output pipeline:{pipeline_id}"
)
```

### With Cluster Execution MCP

Distribute heavy tasks across cluster:

```python
# Route computation-heavy analysis to builder node
result = cluster_execution.offload_to(
    "python analyze_data.py --input large_dataset.csv",
    node_id="builder"
)

# Update task with result
update_task_status(
    task_id=current_task,
    status="completed",
    result=result["output"]
)
```

### With Ember MCP (Quality Enforcement)

Validate outputs meet production standards:

```python
# Before advancing relay
violation = ember.check_violation(
    action="advance_relay",
    params={"output": agent_output},
    context="relay_pipeline"
)

if violation["severity"] > 0.5:
    # Record quality failure
    circuit_breaker_record_failure(
        agent_id=current_agent,
        failure_type="quality_failure",
        error_message=violation["message"]
    )
    retry_relay_step(pipeline_id, current_step)
else:
    advance_relay(...)
```

## Database Schema

```sql
-- Goals table
CREATE TABLE goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    metadata TEXT  -- JSON
);

-- Tasks table
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id INTEGER,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    dependencies TEXT,  -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result TEXT,
    error TEXT,
    metadata TEXT,
    FOREIGN KEY (goal_id) REFERENCES goals(id)
);

-- Task queue for scheduling
CREATE TABLE task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    queue_position INTEGER,
    scheduled_for TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- Indexes for performance
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_goal ON tasks(goal_id);
CREATE INDEX idx_queue_position ON task_queue(queue_position);
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RUNTIME_DB_PATH` | `~/.claude/agent_runtime.db` | Database location |
| `AGENT_RUNTIME_LOG_LEVEL` | `INFO` | Logging verbosity |

### Claude Code Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "agent-runtime": {
      "command": "python",
      "args": ["-m", "agent_runtime_mcp.server"],
      "env": {
        "AGENT_RUNTIME_DB_PATH": "/custom/path/runtime.db"
      }
    }
  }
}
```

## Error Handling

### Task Failures

```python
try:
    # Execute task
    result = execute_task(task)
    update_task_status(task_id, "completed", result=result)
except Exception as e:
    update_task_status(task_id, "failed", error=str(e))
    # Task can be retried or manually resolved
```

### Relay Pipeline Failures

```python
# Single step failure - retry just that step
retry_relay_step(pipeline_id, failed_step_index)

# Cascading failure - circuit breaker trips
if circuit_breaker_status(agent_id)["state"] == "OPEN":
    # Use fallback agent or pause pipeline
    fallback = circuit_breaker_status(agent_id)["fallback_agent"]
```

## Best Practices

1. **Use Goals for Context**: Create goals before tasks to maintain context
2. **Set Dependencies**: Properly define task dependencies to ensure order
3. **Monitor Circuit Breakers**: Check breaker status before heavy operations
4. **Quality Gates**: Use Ember MCP for output validation in relay pipelines
5. **Token Budgeting**: Set realistic token budgets for relay pipelines
6. **Memory Integration**: Store outputs in enhanced-memory for cross-session access

## Files

| File | Purpose |
|------|---------|
| `server.py` | Main MCP server, goal/task management |
| `relay_protocol.py` | Relay Race Protocol implementation |
| `relay_orchestrator.py` | Pipeline orchestration logic |
| `circuit_breaker.py` | Circuit Breaker pattern implementation |

---

*Part of the [Agentic System](https://github.com/marc-shade/agentic-system-oss) MCP ecosystem.*
