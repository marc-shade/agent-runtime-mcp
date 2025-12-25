#!/usr/bin/env python3
"""
Agent Runtime MCP Server
Persistent task queues and goal decomposition for cross-session AGI autonomy.

Provides:
- Persistent task queues that survive sessions
- AI-powered goal decomposition
- Task scheduling and monitoring
- Progress tracking and recovery
- Relay Race Protocol for 48+ agent pipelines (God Agent integration)
"""

import asyncio
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
import mcp.server.stdio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# Initialize Relay Protocol Manager (God Agent Phase 2)
relay_protocol = None
try:
    from relay_protocol import RelayProtocolManager, STANDARD_AGENTS, AgentSpec
    relay_protocol = RelayProtocolManager(DB_PATH)
    logger.info("Relay Race Protocol initialized (God Agent Phase 2: 48-agent pipeline support)")
except ImportError as e:
    logger.warning(f"Relay Protocol not available: {e}")
except Exception as e:
    logger.warning(f"Relay Protocol initialization failed: {e}")

# Initialize Circuit Breaker Registry (God Agent Phase 5)
circuit_registry = None
circuit_executor = None
try:
    from circuit_breaker import CircuitBreakerRegistry, FaultTolerantExecutor
    circuit_registry = CircuitBreakerRegistry(str(DB_PATH))
    circuit_executor = FaultTolerantExecutor(circuit_registry)
    logger.info("Circuit Breaker initialized (God Agent Phase 5: Tiny Dancer fault tolerance)")
except ImportError as e:
    logger.warning(f"Circuit Breaker not available: {e}")
except Exception as e:
    logger.warning(f"Circuit Breaker initialization failed: {e}")


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
        ),
        # ============================================================================
        # RELAY RACE PROTOCOL TOOLS (God Agent Integration - Phase 2)
        # ============================================================================
        Tool(
            name="create_relay_pipeline",
            description="Create a 48-agent relay race pipeline for sequential execution with structured handoffs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Pipeline name"
                    },
                    "goal": {
                        "type": "string",
                        "description": "High-level goal description"
                    },
                    "agent_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Agent types in execution order: researcher, analyzer, synthesizer, validator, formatter, domain_expert"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional detailed description"
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Total token budget for pipeline",
                        "default": 100000
                    }
                },
                "required": ["name", "goal", "agent_types"]
            }
        ),
        Tool(
            name="get_relay_status",
            description="Get current status of a relay pipeline including progress and quality scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline ID"
                    }
                },
                "required": ["pipeline_id"]
            }
        ),
        Tool(
            name="advance_relay",
            description="Manually advance relay to next step after completing current step. Passes the baton.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline ID"
                    },
                    "quality_score": {
                        "type": "number",
                        "description": "Quality score of completed step (0.0-1.0)"
                    },
                    "l_score": {
                        "type": "number",
                        "description": "L-Score of output (0.0-1.0)"
                    },
                    "output_entity_id": {
                        "type": "integer",
                        "description": "Entity ID where output was stored"
                    },
                    "tokens_used": {
                        "type": "integer",
                        "description": "Tokens consumed by step"
                    },
                    "output_summary": {
                        "type": "string",
                        "description": "Summary for next agent"
                    }
                },
                "required": ["pipeline_id", "quality_score", "l_score", "output_entity_id", "tokens_used"]
            }
        ),
        Tool(
            name="retry_relay_step",
            description="Retry a failed relay step without restarting entire pipeline (single-agent retry).",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline ID"
                    },
                    "step_index": {
                        "type": "integer",
                        "description": "Step to retry (0-indexed)"
                    }
                },
                "required": ["pipeline_id", "step_index"]
            }
        ),
        Tool(
            name="list_relay_pipelines",
            description="List relay pipelines with optional status filter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "failed"],
                        "description": "Filter by status"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="get_relay_baton",
            description="Get current baton for a pipeline with all context needed for next agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_id": {
                        "type": "string",
                        "description": "Pipeline ID"
                    }
                },
                "required": ["pipeline_id"]
            }
        ),
        # ============================================================================
        # CIRCUIT BREAKER TOOLS (God Agent Integration - Phase 5: Tiny Dancer)
        # ============================================================================
        Tool(
            name="circuit_breaker_status",
            description="Get circuit breaker status for an agent (CLOSED/OPEN/HALF_OPEN).",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent identifier"
                    }
                },
                "required": ["agent_id"]
            }
        ),
        Tool(
            name="circuit_breaker_list",
            description="List all circuit breakers with their states and open/degraded circuits.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="circuit_breaker_trip",
            description="Manually trip a circuit breaker to OPEN state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent to trip"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for manual trip",
                        "default": "manual intervention"
                    }
                },
                "required": ["agent_id"]
            }
        ),
        Tool(
            name="circuit_breaker_reset",
            description="Reset circuit breaker to CLOSED state.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent to reset"
                    }
                },
                "required": ["agent_id"]
            }
        ),
        Tool(
            name="circuit_breaker_configure",
            description="Configure circuit breaker thresholds for an agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent to configure"
                    },
                    "failure_threshold": {
                        "type": "integer",
                        "description": "Failures before tripping",
                        "default": 5
                    },
                    "window_seconds": {
                        "type": "integer",
                        "description": "Sliding window for failures",
                        "default": 60
                    },
                    "cooldown_seconds": {
                        "type": "integer",
                        "description": "Cooldown before recovery trial",
                        "default": 300
                    },
                    "fallback_agent": {
                        "type": "string",
                        "description": "Agent to use when circuit open",
                        "default": "generalist"
                    }
                },
                "required": ["agent_id"]
            }
        ),
        Tool(
            name="circuit_breaker_record_failure",
            description="Record a failure for circuit breaker tracking.",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent that failed"
                    },
                    "failure_type": {
                        "type": "string",
                        "enum": ["timeout", "exception", "quality_failure", "resource_exhausted", "rate_limited", "invalid_output"],
                        "description": "Type of failure"
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Error description"
                    }
                },
                "required": ["agent_id", "failure_type", "error_message"]
            }
        ),
        Tool(
            name="circuit_breaker_record_success",
            description="Record a success (helps circuit recover from HALF_OPEN to CLOSED).",
            inputSchema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Agent that succeeded"
                    }
                },
                "required": ["agent_id"]
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

    # ============================================================================
    # RELAY RACE PROTOCOL HANDLERS (God Agent Integration - Phase 2)
    # ============================================================================

    elif name == "create_relay_pipeline":
        if not relay_protocol:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Relay Protocol not available"})
            )]

        # Convert agent types to specs
        agents = []
        for agent_type in arguments["agent_types"]:
            if agent_type in STANDARD_AGENTS:
                agents.append(STANDARD_AGENTS[agent_type])
            else:
                # Create custom agent
                agents.append(AgentSpec(
                    agent_id=agent_type,
                    agent_type=agent_type,
                    task_template=f"Execute {agent_type} task based on input context.",
                    expected_output_type="analysis",
                    max_tokens=4000
                ))

        pipeline = relay_protocol.create_pipeline(
            name=arguments["name"],
            goal=arguments["goal"],
            agents=agents,
            description=arguments.get("description", ""),
            token_budget=arguments.get("token_budget", 100000)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "goal": pipeline.goal,
                "agent_count": len(agents),
                "agents": [a.agent_type for a in agents],
                "token_budget": arguments.get("token_budget", 100000)
            }, indent=2)
        )]

    elif name == "get_relay_status":
        if not relay_protocol:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Relay Protocol not available"})
            )]

        pipeline = relay_protocol.get_pipeline(arguments["pipeline_id"])
        if not pipeline:
            return [TextContent(
                type="text",
                text=json.dumps({"success": False, "error": f"Pipeline {arguments['pipeline_id']} not found"})
            )]

        return [TextContent(
            type="text",
            text=json.dumps({"success": True, **pipeline}, indent=2)
        )]

    elif name == "advance_relay":
        if not relay_protocol:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Relay Protocol not available"})
            )]

        current_baton = relay_protocol.get_current_baton(arguments["pipeline_id"])
        if not current_baton:
            return [TextContent(
                type="text",
                text=json.dumps({"success": False, "error": "No active baton found for pipeline"})
            )]

        next_baton = relay_protocol.pass_baton(
            baton_id=current_baton.baton_id,
            quality_score=arguments["quality_score"],
            l_score=arguments["l_score"],
            output_entity_id=arguments["output_entity_id"],
            tokens_used=arguments["tokens_used"],
            output_summary=arguments.get("output_summary", "")
        )

        if next_baton is None:
            # Check if completed or failed
            pipeline = relay_protocol.get_pipeline(arguments["pipeline_id"])
            if pipeline and pipeline.get("status") == "completed":
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "status": "pipeline_completed",
                        "message": "All steps completed successfully"
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "status": "quality_gate_failed",
                        "quality_score": arguments["quality_score"],
                        "l_score": arguments["l_score"]
                    }, indent=2)
                )]

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "next_baton": next_baton.to_dict(),
                "next_step": next_baton.sequence_position,
                "total_steps": next_baton.total_agents
            }, indent=2)
        )]

    elif name == "retry_relay_step":
        if not relay_protocol:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Relay Protocol not available"})
            )]

        new_baton = relay_protocol.retry_step(
            arguments["pipeline_id"],
            arguments["step_index"]
        )

        if not new_baton:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": f"Cannot retry step {arguments['step_index']}: max retries exceeded"
                })
            )]

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "baton": new_baton.to_dict(),
                "retry_count": new_baton.retry_count
            }, indent=2)
        )]

    elif name == "list_relay_pipelines":
        if not relay_protocol:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Relay Protocol not available"})
            )]

        pipelines = relay_protocol.list_pipelines(
            status=arguments.get("status"),
            limit=arguments.get("limit", 50)
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "count": len(pipelines),
                "pipelines": pipelines
            }, indent=2)
        )]

    elif name == "get_relay_baton":
        if not relay_protocol:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Relay Protocol not available"})
            )]

        baton = relay_protocol.get_current_baton(arguments["pipeline_id"])

        if not baton:
            return [TextContent(
                type="text",
                text=json.dumps({"success": False, "error": "No active baton found"})
            )]

        # Generate agent prompt for current step
        from relay_protocol import generate_agent_prompt
        prompt = generate_agent_prompt(
            baton,
            AgentSpec(
                agent_id="current",
                agent_type="agent",
                task_template=baton.task_description,
                expected_output_type=baton.expected_output_type
            )
        )

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "baton": baton.to_dict(),
                "prompt": prompt
            }, indent=2)
        )]

    # ============================================================================
    # CIRCUIT BREAKER HANDLERS (God Agent Phase 5: Tiny Dancer)
    # ============================================================================

    elif name == "circuit_breaker_status":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        breaker = circuit_registry.get_or_create(arguments["agent_id"])
        return [TextContent(
            type="text",
            text=json.dumps(breaker.get_status(), indent=2, default=str)
        )]

    elif name == "circuit_breaker_list":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        status = circuit_registry.get_all_status()
        summary = {
            "total_breakers": len(status),
            "open_circuits": circuit_registry.get_open_circuits(),
            "half_open_circuits": circuit_registry.get_half_open_circuits(),
            "breakers": status
        }
        return [TextContent(
            type="text",
            text=json.dumps(summary, indent=2, default=str)
        )]

    elif name == "circuit_breaker_trip":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        breaker = circuit_registry.get_or_create(arguments["agent_id"])
        old_state = breaker.current_state.value
        reason = arguments.get("reason", "manual intervention")
        breaker.force_open(reason)
        circuit_registry._persist_breaker(arguments["agent_id"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "agent_id": arguments["agent_id"],
                "action": "trip",
                "old_state": old_state,
                "new_state": "open",
                "reason": reason,
                "cooldown_until": breaker.state.cooldown_until.isoformat() if breaker.state.cooldown_until else None
            }, indent=2)
        )]

    elif name == "circuit_breaker_reset":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        breaker = circuit_registry.get_or_create(arguments["agent_id"])
        old_state = breaker.current_state.value
        breaker.force_close()
        circuit_registry._persist_breaker(arguments["agent_id"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "agent_id": arguments["agent_id"],
                "action": "reset",
                "old_state": old_state,
                "new_state": "closed"
            }, indent=2)
        )]

    elif name == "circuit_breaker_configure":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        config = {
            "failure_threshold": arguments.get("failure_threshold", 5),
            "window_seconds": arguments.get("window_seconds", 60),
            "cooldown_seconds": arguments.get("cooldown_seconds", 300),
            "fallback_agent": arguments.get("fallback_agent", "generalist"),
            "half_open_max_calls": 3,
            "success_threshold": 2
        }

        breaker = circuit_registry.get_or_create(arguments["agent_id"], config)
        breaker.state.config.update(config)
        circuit_registry._persist_breaker(arguments["agent_id"])

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "agent_id": arguments["agent_id"],
                "config": config
            }, indent=2)
        )]

    elif name == "circuit_breaker_record_failure":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        from circuit_breaker import FailureType
        breaker = circuit_registry.get_or_create(arguments["agent_id"])

        try:
            ftype = FailureType(arguments["failure_type"])
        except ValueError:
            ftype = FailureType.EXCEPTION

        old_state = breaker.current_state.value
        breaker.record_failure(ftype, arguments["error_message"])
        circuit_registry._persist_breaker(arguments["agent_id"])
        new_state = breaker.current_state.value

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "agent_id": arguments["agent_id"],
                "failure_recorded": True,
                "old_state": old_state,
                "new_state": new_state,
                "tripped": old_state != new_state
            }, indent=2, default=str)
        )]

    elif name == "circuit_breaker_record_success":
        if not circuit_registry:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Circuit Breaker not available"})
            )]

        breaker = circuit_registry.get_or_create(arguments["agent_id"])
        old_state = breaker.current_state.value
        breaker.record_success()
        circuit_registry._persist_breaker(arguments["agent_id"])
        new_state = breaker.current_state.value

        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "agent_id": arguments["agent_id"],
                "success_recorded": True,
                "old_state": old_state,
                "new_state": new_state,
                "recovered": old_state == "half_open" and new_state == "closed"
            }, indent=2, default=str)
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
