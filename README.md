# Agent Runtime MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Part of Agentic System](https://img.shields.io/badge/Part_of-Agentic_System-brightgreen)](https://github.com/marc-shade/agentic-system-oss)

> **Persistent task queues and goal decomposition for cross-session AGI autonomy.**

Part of the [Agentic System](https://github.com/marc-shade/agentic-system-oss) - a 24/7 autonomous AI framework with persistent memory and self-improvement.

## Features

- **Persistent Task Queues**: Tasks survive across Claude Code sessions
- **Goal Decomposition**: AI-powered breakdown of complex goals into tasks
- **Task Scheduling**: Priority-based task management with dependencies
- **Progress Tracking**: Monitor task completion and recovery from failures
- **Relay Race Protocol**: Multi-agent pipelines with structured handoffs
- **Circuit Breaker Pattern**: Fault tolerance with automatic recovery

## MCP Tools

### Goal & Task Management

| Tool | Description |
|------|-------------|
| `create_goal` | Create a new goal with name and description |
| `decompose_goal` | Break down a goal into actionable tasks using AI |
| `create_task` | Manually create a task with priority and dependencies |
| `get_next_task` | Get the highest priority task ready for execution |
| `update_task_status` | Update task status (pending/in_progress/completed/failed) |
| `list_goals` | List all goals, optionally filtered by status |
| `list_tasks` | List tasks, optionally filtered by goal or status |
| `get_goal` | Get goal details by ID |
| `get_task` | Get task details by ID |

### Relay Race Protocol

The Relay Race Protocol enables multi-agent pipelines with structured handoffs - like a relay race where each agent passes a "baton" to the next.

```
[Researcher] → [Analyzer] → [Synthesizer] → [Validator] → [Formatter] → [Expert]
     ↓              ↓             ↓              ↓             ↓            ↓
  Research      Analyze       Combine        Verify        Polish      Finalize
  Context       Findings      Insights       Quality       Output      Delivery
```

| Tool | Description |
|------|-------------|
| `create_relay_pipeline` | Create a multi-agent relay pipeline with defined agent sequence |
| `get_relay_status` | Get current status including progress and quality scores |
| `advance_relay` | Pass the baton to next agent with quality metrics |
| `retry_relay_step` | Retry a failed step without restarting entire pipeline |
| `list_relay_pipelines` | List pipelines with optional status filter |
| `get_relay_baton` | Get current baton with all context for next agent |

**Example Usage:**

```python
# Create a research pipeline
pipeline = create_relay_pipeline(
    name="Research Analysis",
    goal="Analyze competitor pricing strategies",
    agent_types=["researcher", "analyzer", "synthesizer", "validator"],
    token_budget=100000
)

# After completing a step, pass the baton
advance_relay(
    pipeline_id=pipeline["pipeline_id"],
    quality_score=0.85,
    l_score=0.9,  # Provenance tracking
    output_entity_id=123,
    tokens_used=5000,
    output_summary="Found 5 competitors with detailed pricing data"
)
```

**Each baton pass includes:**
- Quality score from previous step
- L-Score (provenance tracking)
- Output summary for next agent
- Token budget management

### Circuit Breaker Pattern

The Circuit Breaker prevents cascading failures by temporarily blocking requests to failing agents.

```
CLOSED (normal) → failures exceed threshold → OPEN (blocking)
                                                    ↓
                                              cooldown period
                                                    ↓
                                              HALF_OPEN (testing)
                                                    ↓
                                    success → CLOSED | failure → OPEN
```

| Tool | Description |
|------|-------------|
| `circuit_breaker_status` | Get circuit state for an agent (CLOSED/OPEN/HALF_OPEN) |
| `circuit_breaker_list` | List all circuit breakers with their states |
| `circuit_breaker_trip` | Manually trip a circuit to OPEN state |
| `circuit_breaker_reset` | Reset circuit to CLOSED state |
| `circuit_breaker_configure` | Configure thresholds for an agent |
| `circuit_breaker_record_failure` | Record a failure for tracking |
| `circuit_breaker_record_success` | Record success (helps recovery) |

**Configuration Options:**

```python
circuit_breaker_configure(
    agent_id="researcher",
    failure_threshold=5,      # Failures before tripping (default: 5)
    window_seconds=60,        # Sliding window for failures (default: 60)
    cooldown_seconds=300,     # Time before recovery trial (default: 300)
    fallback_agent="generalist"  # Agent to use when circuit open
)
```

**Failure Types:**
- `timeout` - Agent took too long
- `exception` - Agent threw an error
- `quality_failure` - Output didn't meet quality threshold
- `resource_exhausted` - Token budget exceeded
- `rate_limited` - Hit API rate limits
- `invalid_output` - Output format invalid

## Requirements

- Python 3.10+
- mcp SDK
- SQLite (built-in)

## Installation

```bash
pip install mcp
```

## Usage

```bash
python server.py
```

## Data Storage

Goals and tasks are stored in `~/.claude/agent_runtime.db` (SQLite).

## Integration

This MCP enables AGI systems to maintain continuity across sessions by persisting goals and tasks.

## Part of the MCP Ecosystem

This server integrates with other MCP servers for comprehensive AGI capabilities:

| Server | Purpose |
|--------|---------|
| [enhanced-memory-mcp](https://github.com/marc-shade/enhanced-memory-mcp) | 4-tier persistent memory with semantic search |
| [cluster-execution-mcp](https://github.com/marc-shade/cluster-execution-mcp) | Distributed execution across clusters |
| [node-chat-mcp](https://github.com/marc-shade/node-chat-mcp) | Inter-node AI communication |
| [ember-mcp](https://github.com/marc-shade/ember-mcp) | Production-only policy enforcement |
| [research-paper-mcp](https://github.com/marc-shade/research-paper-mcp) | arXiv/Semantic Scholar paper search |

See [agentic-system-oss](https://github.com/marc-shade/agentic-system-oss) for the complete framework with all MCP servers pre-configured.

## License

MIT
