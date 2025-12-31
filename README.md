# Agent Runtime MCP

Persistent task queues and goal decomposition for cross-session AGI autonomy with God Agent integration.

## Description

Agent Runtime MCP provides persistent task management that survives across sessions, enabling true autonomous AGI workflows. Features include:

- **Persistent Goals & Tasks**: SQLite-backed storage that survives restarts
- **AI-Powered Goal Decomposition**: Break complex goals into executable tasks
- **Dependency Management**: Automatic task ordering based on dependencies
- **Priority Queuing**: Intelligent task scheduling by priority and readiness
- **Relay Race Protocol** (God Agent Phase 2): 48-agent pipelines with structured handoffs
- **Circuit Breaker** (God Agent Phase 5): Fault tolerance with automatic fallback
- **Cross-Session Continuity**: Resume work exactly where you left off

## Installation

### Using pip

```bash
git clone https://github.com/marc-shade/agent-runtime-mcp
cd agent-runtime-mcp
pip install -r requirements.txt
```

### Using uv (recommended)

```bash
git clone https://github.com/marc-shade/agent-runtime-mcp
cd agent-runtime-mcp
uv pip install -r requirements.txt
```

### Dependencies

```bash
pip install anthropic-mcp
```

## Configuration

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "agent-runtime": {
      "command": "python3",
      "args": [
        "/absolute/path/to/agent-runtime-mcp/server.py"
      ]
    }
  }
}
```

## Tools

### Core Goal & Task Management (9)

| Tool | Description |
|------|-------------|
| `create_goal` | Create high-level goal with name and description |
| `decompose_goal` | Use AI to break goal into tasks (sequential/parallel/hierarchical) |
| `create_task` | Manually create task with dependencies |
| `get_next_task` | Get next ready task from queue (highest priority, deps met) |
| `update_task_status` | Update status (pending/in_progress/completed/failed/cancelled) |
| `list_goals` | List all goals, optionally filtered by status |
| `list_tasks` | List tasks by goal, status, with limit |
| `get_goal` | Get goal details by ID |
| `get_task` | Get task details by ID |

### Relay Race Protocol (God Agent Phase 2) (6)

| Tool | Description |
|------|-------------|
| `create_relay_pipeline` | Create 48-agent relay race with baton passing |
| `get_relay_status` | Get pipeline status (progress, quality scores) |
| `advance_relay` | Pass baton to next agent after completing step |
| `retry_relay_step` | Retry failed step without restarting pipeline |
| `list_relay_pipelines` | List pipelines by status |
| `get_relay_baton` | Get current baton with context for next agent |

### Circuit Breaker (God Agent Phase 5: Tiny Dancer) (7)

| Tool | Description |
|------|-------------|
| `circuit_breaker_status` | Get breaker state (CLOSED/OPEN/HALF_OPEN) |
| `circuit_breaker_list` | List all breakers with open/degraded circuits |
| `circuit_breaker_trip` | Manually trip breaker to OPEN state |
| `circuit_breaker_reset` | Reset breaker to CLOSED state |
| `circuit_breaker_configure` | Configure thresholds (failures, window, cooldown) |
| `circuit_breaker_record_failure` | Record failure for tracking |
| `circuit_breaker_record_success` | Record success (helps recovery) |

## Usage Examples

### Basic Goal Creation

```python
# Create goal
goal = mcp__agent-runtime__create_goal({
    "name": "Build REST API",
    "description": "Create RESTful API for user authentication with JWT tokens",
    "metadata": {"priority": "high", "project": "auth-service"}
})
# Returns: {"id": 1, "name": "Build REST API", "status": "active", ...}
```

### AI Goal Decomposition

```python
# Decompose goal into tasks (sequential strategy)
result = mcp__agent-runtime__decompose_goal({
    "goal_id": 1,
    "strategy": "sequential"
})
# Returns: {
#   "goal_id": 1,
#   "strategy": "sequential",
#   "tasks_created": [101, 102, 103, 104, 105],
#   "count": 5
# }
# Tasks: Research → Plan → Implement → Test → Document (with dependencies)
```

### Parallel Decomposition

```python
# Decompose for parallel execution
result = mcp__agent-runtime__decompose_goal({
    "goal_id": 1,
    "strategy": "parallel"
})
# Creates: Backend, Frontend, Testing tasks (no dependencies, run simultaneously)
```

### Hierarchical Decomposition

```python
# Decompose into phases
result = mcp__agent-runtime__decompose_goal({
    "goal_id": 1,
    "strategy": "hierarchical"
})
# Creates: Phase 1 (Foundation) → Phase 2 (Core) → Phase 3 (Integration) → Phase 4 (Optimization)
```

### Task Queue Processing

```python
# Get next ready task
task = mcp__agent-runtime__get_next_task()
# Returns: Highest priority task with all dependencies met
# {"id": 101, "title": "Research requirements...", "priority": 10, ...}

# Start work
mcp__agent-runtime__update_task_status({
    "task_id": 101,
    "status": "in_progress"
})

# Complete task
mcp__agent-runtime__update_task_status({
    "task_id": 101,
    "status": "completed",
    "result": "Requirements documented in docs/api-spec.md"
})

# Get next (automatically handles dependencies)
next_task = mcp__agent-runtime__get_next_task()
# Returns: Task 102 (Plan approach) since Research (101) is complete
```

### Manual Task Creation with Dependencies

```python
# Create task with explicit dependencies
mcp__agent-runtime__create_task({
    "goal_id": 1,
    "title": "Deploy to production",
    "description": "Deploy authentication service",
    "priority": 7,
    "dependencies": [103, 104]  # Wait for Implementation and Testing
})
```

### Relay Race Pipeline (48-Agent)

```python
# Create relay pipeline for complex workflow
pipeline = mcp__agent-runtime__create_relay_pipeline({
    "name": "Research Paper Analysis",
    "goal": "Extract insights from 10 AGI papers",
    "agent_types": [
        "researcher",      # Gather papers
        "analyzer",        # Extract key points
        "synthesizer",     # Find patterns
        "validator",       # Check quality
        "formatter"        # Create report
    ],
    "token_budget": 100000
})
# Returns: {"pipeline_id": "rp_abc123", "agent_count": 5, ...}

# Check pipeline status
status = mcp__agent-runtime__get_relay_status({
    "pipeline_id": "rp_abc123"
})
# Returns: {
#   "current_step": 2,
#   "total_steps": 5,
#   "status": "in_progress",
#   "quality_scores": [0.92, 0.88, ...],
#   "tokens_used": 24531
# }

# Get current baton (context for next agent)
baton = mcp__agent-runtime__get_relay_baton({
    "pipeline_id": "rp_abc123"
})
# Returns: {
#   "baton": {...},
#   "prompt": "You are the Synthesizer. Previous output: ..."
# }

# Advance to next step
mcp__agent-runtime__advance_relay({
    "pipeline_id": "rp_abc123",
    "quality_score": 0.88,
    "l_score": 0.85,
    "output_entity_id": 456,
    "tokens_used": 8234,
    "output_summary": "Found 3 key patterns across papers"
})

# Retry failed step
mcp__agent-runtime__retry_relay_step({
    "pipeline_id": "rp_abc123",
    "step_index": 2
})
```

### Circuit Breaker (Fault Tolerance)

```python
# Check agent circuit breaker status
status = mcp__agent-runtime__circuit_breaker_status({
    "agent_id": "researcher_agent"
})
# Returns: {
#   "agent_id": "researcher_agent",
#   "state": "CLOSED",
#   "failure_count": 0,
#   "success_count": 42
# }

# Record failure
mcp__agent-runtime__circuit_breaker_record_failure({
    "agent_id": "researcher_agent",
    "failure_type": "timeout",
    "error_message": "API request timed out after 30s"
})

# List all circuit breakers
breakers = mcp__agent-runtime__circuit_breaker_list()
# Returns: {
#   "total_breakers": 10,
#   "open_circuits": ["failing_agent_1", "failing_agent_2"],
#   "half_open_circuits": ["recovering_agent"],
#   "breakers": [...]
# }

# Configure thresholds
mcp__agent-runtime__circuit_breaker_configure({
    "agent_id": "researcher_agent",
    "failure_threshold": 5,
    "window_seconds": 60,
    "cooldown_seconds": 300,
    "fallback_agent": "generalist"
})

# Manually trip (emergency stop)
mcp__agent-runtime__circuit_breaker_trip({
    "agent_id": "researcher_agent",
    "reason": "Manual intervention - debugging required"
})

# Reset after fix
mcp__agent-runtime__circuit_breaker_reset({
    "agent_id": "researcher_agent"
})
```

### Cross-Session Resume

```python
# Session 1: Create goal and start work
goal = mcp__agent-runtime__create_goal({"name": "Big Project", ...})
mcp__agent-runtime__decompose_goal({"goal_id": goal["id"]})
task1 = mcp__agent-runtime__get_next_task()
mcp__agent-runtime__update_task_status({"task_id": task1["id"], "status": "in_progress"})

# [Close Claude Code, restart later]

# Session 2: Resume exactly where left off
pending = mcp__agent-runtime__list_tasks({"status": "in_progress"})
# Returns: [task1] - still marked as in_progress
task1_updated = mcp__agent-runtime__update_task_status({
    "task_id": task1["id"],
    "status": "completed"
})
next_task = mcp__agent-runtime__get_next_task()
# Automatically gets task2 (next in dependency chain)
```

## Requirements

- **Python**: 3.10+
- **Dependencies**: `anthropic-mcp` (MCP SDK)
- **Storage**: `~/.claude/agent_runtime.db` (SQLite)

## Database Schema

Tables in `~/.claude/agent_runtime.db`:

- `goals` - High-level goals with status and metadata
- `tasks` - Individual tasks with dependencies, priority, results
- `task_queue` - Queue position and scheduling info
- `relay_pipelines` - Relay race pipeline definitions (God Agent Phase 2)
- `relay_batons` - Baton state for pipeline steps
- `circuit_breakers` - Circuit breaker state and history (God Agent Phase 5)

## Decomposition Strategies

### Sequential
```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5
```
Each task depends on previous. Linear execution.

### Parallel
```
Task 1 (Backend)  ─┐
Task 2 (Frontend) ─┼─→ All run simultaneously
Task 3 (Testing)  ─┘
```
No dependencies. Maximum parallelism.

### Hierarchical
```
Phase 1 (Foundation)
  ↓
Phase 2 (Core Implementation)
  ↓
Phase 3 (Integration)
  ↓
Phase 4 (Optimization)
```
Large phases that can be further decomposed.

## Testing

```bash
# Run test suite
python3 test_agent_runtime.py

# Test relay protocol
python3 test_relay_protocol.py

# Test circuit breaker
python3 test_circuit_breaker.py
```

## God Agent Integration

### Phase 2: Relay Race Protocol
- 48-agent sequential pipelines
- Structured baton passing with context
- Quality gates at each step
- L-Score tracking for output quality
- Single-step retry (no full restart)

### Phase 5: Circuit Breaker (Tiny Dancer)
- Automatic failure detection
- State machine: CLOSED → OPEN → HALF_OPEN → CLOSED
- Configurable thresholds and cooldowns
- Fallback agent routing
- Recovery monitoring

## Links

- GitHub: https://github.com/marc-shade/agent-runtime-mcp
- Issues: https://github.com/marc-shade/agent-runtime-mcp/issues
