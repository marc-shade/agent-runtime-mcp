#!/usr/bin/env python3
"""
Agent Runtime MCP Server
Persistent task queues and goal decomposition for cross-session AGI autonomy.

Provides:
- Persistent task queues that survive sessions
- AI-powered goal decomposition
- Task scheduling and monitoring
- Progress tracking and recovery
"""

import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.server.stdio


# Database path
DB_PATH = Path.home() / ".claude" / "agent_runtime.db"
DB_PATH.parent.mkdir(exist_ok=True)


class AgentRuntimeDB:
    """Database manager for persistent task and goal storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_id INTEGER,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    dependencies TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result TEXT,
                    error TEXT,
                    metadata TEXT,
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id INTEGER NOT NULL,
                    queue_position INTEGER,
                    scheduled_for TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_position ON task_queue(queue_position)
            """)

            conn.commit()

    def create_goal(self, name: str, description: str, metadata: Dict = None) -> int:
        """Create a new goal."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO goals (name, description, metadata) VALUES (?, ?, ?)",
                (name, description, json.dumps(metadata or {}))
            )
            conn.commit()
            return cursor.lastrowid

    def get_goal(self, goal_id: int) -> Optional[Dict]:
        """Get goal by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def list_goals(self, status: Optional[str] = None) -> List[Dict]:
        """List all goals, optionally filtered by status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cursor = conn.execute("SELECT * FROM goals WHERE status = ? ORDER BY created_at DESC", (status,))
            else:
                cursor = conn.execute("SELECT * FROM goals ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def update_goal_status(self, goal_id: int, status: str):
        """Update goal status."""
        with sqlite3.connect(self.db_path) as conn:
            completed_at = datetime.now().isoformat() if status == 'completed' else None
            conn.execute(
                "UPDATE goals SET status = ?, updated_at = ?, completed_at = ? WHERE id = ?",
                (status, datetime.now().isoformat(), completed_at, goal_id)
            )
            conn.commit()

    def create_task(
        self,
        goal_id: Optional[int],
        title: str,
        description: str = "",
        priority: int = 5,
        dependencies: List[int] = None,
        metadata: Dict = None
    ) -> int:
        """Create a new task."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """INSERT INTO tasks
                   (goal_id, title, description, priority, dependencies, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    goal_id,
                    title,
                    description,
                    priority,
                    json.dumps(dependencies or []),
                    json.dumps(metadata or {})
                )
            )
            task_id = cursor.lastrowid

            # Add to queue
            conn.execute(
                "INSERT INTO task_queue (task_id, queue_position) VALUES (?, ?)",
                (task_id, self._get_next_queue_position(conn))
            )

            conn.commit()
            return task_id

    def _get_next_queue_position(self, conn) -> int:
        """Get next available queue position."""
        cursor = conn.execute("SELECT MAX(queue_position) FROM task_queue")
        max_pos = cursor.fetchone()[0]
        return (max_pos or 0) + 1

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Get task by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                task = dict(row)
                task['dependencies'] = json.loads(task['dependencies'])
                task['metadata'] = json.loads(task['metadata'] or '{}')
                return task
        return None

    def list_tasks(
        self,
        goal_id: Optional[int] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List tasks with optional filters."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM tasks WHERE 1=1"
            params = []

            if goal_id is not None:
                query += " AND goal_id = ?"
                params.append(goal_id)

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            tasks = []
            for row in cursor.fetchall():
                task = dict(row)
                task['dependencies'] = json.loads(task['dependencies'])
                task['metadata'] = json.loads(task['metadata'] or '{}')
                tasks.append(task)

            return tasks

    def get_next_task(self) -> Optional[Dict]:
        """Get next task from queue (highest priority, no unmet dependencies)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get tasks in queue order, considering priority
            cursor = conn.execute("""
                SELECT t.* FROM tasks t
                INNER JOIN task_queue q ON t.id = q.task_id
                WHERE t.status = 'pending'
                ORDER BY t.priority DESC, q.queue_position ASC
                LIMIT 50
            """)

            for row in cursor.fetchall():
                task = dict(row)
                task['dependencies'] = json.loads(task['dependencies'])
                task['metadata'] = json.loads(task['metadata'] or '{}')

                # Check if dependencies are met
                if self._dependencies_met(conn, task['dependencies']):
                    return task

            return None

    def _dependencies_met(self, conn, dependencies: List[int]) -> bool:
        """Check if all task dependencies are completed."""
        if not dependencies:
            return True

        placeholders = ','.join('?' * len(dependencies))
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM tasks WHERE id IN ({placeholders}) AND status != 'completed'",
            dependencies
        )
        unmet = cursor.fetchone()[0]
        return unmet == 0

    def update_task_status(
        self,
        task_id: int,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update task status and optionally store result or error."""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now().isoformat()

            started_at = now if status == 'in_progress' else None
            completed_at = now if status in ('completed', 'failed') else None

            conn.execute("""
                UPDATE tasks
                SET status = ?,
                    updated_at = ?,
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?,
                    result = ?,
                    error = ?
                WHERE id = ?
            """, (status, now, started_at, completed_at, result, error, task_id))

            # Remove from queue if completed or failed
            if status in ('completed', 'failed', 'cancelled'):
                conn.execute("DELETE FROM task_queue WHERE task_id = ?", (task_id,))

            conn.commit()


class GoalDecomposer:
    """AI-powered goal decomposition into tasks."""

    @staticmethod
    async def decompose_goal(goal: Dict, strategy: str = "sequential") -> List[Dict]:
        """
        Decompose a goal into tasks using AI.

        Strategies:
        - sequential: Tasks must run in order
        - parallel: Tasks can run simultaneously
        - hierarchical: Break into sub-goals and tasks
        """

        # This would use an LLM to intelligently break down goals
        # For now, we'll create a simple decomposition

        tasks = []

        if strategy == "sequential":
            tasks = [
                {
                    "title": f"Research requirements for: {goal['name']}",
                    "description": f"Gather information needed for {goal['description']}",
                    "priority": 10,
                    "dependencies": []
                },
                {
                    "title": f"Plan approach for: {goal['name']}",
                    "description": "Create detailed implementation plan",
                    "priority": 9,
                    "dependencies": []
                },
                {
                    "title": f"Implement: {goal['name']}",
                    "description": goal['description'],
                    "priority": 8,
                    "dependencies": []
                },
                {
                    "title": f"Test: {goal['name']}",
                    "description": "Verify implementation works correctly",
                    "priority": 7,
                    "dependencies": []
                },
                {
                    "title": f"Document: {goal['name']}",
                    "description": "Create documentation for implementation",
                    "priority": 6,
                    "dependencies": []
                }
            ]

        elif strategy == "parallel":
            # Break into parallel workstreams
            tasks = [
                {
                    "title": f"Backend work for: {goal['name']}",
                    "description": f"Backend implementation: {goal['description']}",
                    "priority": 10,
                    "dependencies": []
                },
                {
                    "title": f"Frontend work for: {goal['name']}",
                    "description": f"Frontend implementation: {goal['description']}",
                    "priority": 10,
                    "dependencies": []
                },
                {
                    "title": f"Testing for: {goal['name']}",
                    "description": "Comprehensive testing",
                    "priority": 10,
                    "dependencies": []
                }
            ]

        elif strategy == "hierarchical":
            # Create sub-goals that can be further decomposed
            tasks = [
                {
                    "title": f"Phase 1: Foundation for {goal['name']}",
                    "description": "Set up basic infrastructure",
                    "priority": 10,
                    "dependencies": []
                },
                {
                    "title": f"Phase 2: Core implementation for {goal['name']}",
                    "description": "Build main functionality",
                    "priority": 9,
                    "dependencies": []
                },
                {
                    "title": f"Phase 3: Integration for {goal['name']}",
                    "description": "Integrate with existing systems",
                    "priority": 8,
                    "dependencies": []
                },
                {
                    "title": f"Phase 4: Optimization for {goal['name']}",
                    "description": "Performance tuning and polish",
                    "priority": 7,
                    "dependencies": []
                }
            ]

        return tasks


# Initialize database
db = AgentRuntimeDB(DB_PATH)
decomposer = GoalDecomposer()


# MCP Server
app = Server("agent-runtime-mcp")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available agent runtime tools."""
    return [
        Tool(
            name="create_goal",
            description="Create a new goal with name and description. Goals persist across sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name for the goal"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what the goal aims to achieve"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata (tags, context, etc.)"
                    }
                },
                "required": ["name", "description"]
            }
        ),
        Tool(
            name="decompose_goal",
            description="Decompose a goal into tasks using AI. Returns task IDs that were created.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "integer",
                        "description": "ID of the goal to decompose"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["sequential", "parallel", "hierarchical"],
                        "description": "Decomposition strategy",
                        "default": "sequential"
                    }
                },
                "required": ["goal_id"]
            }
        ),
        Tool(
            name="create_task",
            description="Create a new task manually. Tasks persist in queue across sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "integer",
                        "description": "Optional goal ID to associate task with"
                    },
                    "title": {
                        "type": "string",
                        "description": "Task title"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed task description"
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority (1-10, 10 is highest)",
                        "default": 5
                    },
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Task IDs that must complete before this task"
                    }
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="get_next_task",
            description="Get the next task from the queue (highest priority, dependencies met).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="update_task_status",
            description="Update task status (pending, in_progress, completed, failed, cancelled).",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "Task ID to update"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed", "cancelled"],
                        "description": "New status"
                    },
                    "result": {
                        "type": "string",
                        "description": "Optional result data (for completed tasks)"
                    },
                    "error": {
                        "type": "string",
                        "description": "Optional error message (for failed tasks)"
                    }
                },
                "required": ["task_id", "status"]
            }
        ),
        Tool(
            name="list_goals",
            description="List all goals, optionally filtered by status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["active", "completed", "cancelled"],
                        "description": "Optional status filter"
                    }
                }
            }
        ),
        Tool(
            name="list_tasks",
            description="List tasks, optionally filtered by goal or status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "integer",
                        "description": "Optional goal ID filter"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed", "cancelled"],
                        "description": "Optional status filter"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of tasks to return",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="get_goal",
            description="Get goal details by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "integer",
                        "description": "Goal ID"
                    }
                },
                "required": ["goal_id"]
            }
        ),
        Tool(
            name="get_task",
            description="Get task details by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "Task ID"
                    }
                },
                "required": ["task_id"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle tool calls."""

    if name == "create_goal":
        goal_id = db.create_goal(
            arguments["name"],
            arguments["description"],
            arguments.get("metadata")
        )
        goal = db.get_goal(goal_id)
        return [TextContent(
            type="text",
            text=json.dumps(goal, indent=2)
        )]

    elif name == "decompose_goal":
        goal = db.get_goal(arguments["goal_id"])
        if not goal:
            return [TextContent(type="text", text=f"Goal {arguments['goal_id']} not found")]

        strategy = arguments.get("strategy", "sequential")
        tasks = await decomposer.decompose_goal(goal, strategy)

        created_task_ids = []
        for i, task in enumerate(tasks):
            # Set up dependencies for sequential strategy
            dependencies = []
            if strategy == "sequential" and i > 0:
                dependencies = [created_task_ids[-1]]

            task_id = db.create_task(
                goal_id=goal["id"],
                title=task["title"],
                description=task["description"],
                priority=task.get("priority", 5),
                dependencies=dependencies
            )
            created_task_ids.append(task_id)

        return [TextContent(
            type="text",
            text=json.dumps({
                "goal_id": goal["id"],
                "strategy": strategy,
                "tasks_created": created_task_ids,
                "count": len(created_task_ids)
            }, indent=2)
        )]

    elif name == "create_task":
        task_id = db.create_task(
            goal_id=arguments.get("goal_id"),
            title=arguments["title"],
            description=arguments.get("description", ""),
            priority=arguments.get("priority", 5),
            dependencies=arguments.get("dependencies")
        )
        task = db.get_task(task_id)
        return [TextContent(
            type="text",
            text=json.dumps(task, indent=2)
        )]

    elif name == "get_next_task":
        task = db.get_next_task()
        if task:
            return [TextContent(
                type="text",
                text=json.dumps(task, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"message": "No tasks available in queue"})
            )]

    elif name == "update_task_status":
        db.update_task_status(
            arguments["task_id"],
            arguments["status"],
            arguments.get("result"),
            arguments.get("error")
        )
        task = db.get_task(arguments["task_id"])
        return [TextContent(
            type="text",
            text=json.dumps(task, indent=2)
        )]

    elif name == "list_goals":
        goals = db.list_goals(arguments.get("status"))
        return [TextContent(
            type="text",
            text=json.dumps(goals, indent=2)
        )]

    elif name == "list_tasks":
        tasks = db.list_tasks(
            goal_id=arguments.get("goal_id"),
            status=arguments.get("status"),
            limit=arguments.get("limit", 50)
        )
        return [TextContent(
            type="text",
            text=json.dumps(tasks, indent=2)
        )]

    elif name == "get_goal":
        goal = db.get_goal(arguments["goal_id"])
        if goal:
            return [TextContent(
                type="text",
                text=json.dumps(goal, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Goal {arguments['goal_id']} not found"})
            )]

    elif name == "get_task":
        task = db.get_task(arguments["task_id"])
        if task:
            return [TextContent(
                type="text",
                text=json.dumps(task, indent=2)
            )]
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Task {arguments['task_id']} not found"})
            )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
