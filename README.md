# Agent Runtime MCP Server

Persistent task queues and goal decomposition for cross-session AGI autonomy.

## Features

- **Persistent Task Queues**: Tasks survive across Claude Code sessions
- **Goal Decomposition**: AI-powered breakdown of complex goals into tasks
- **Task Scheduling**: Priority-based task management with dependencies
- **Progress Tracking**: Monitor task completion and recovery from failures

## MCP Tools

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

## License

MIT
