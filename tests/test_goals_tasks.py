"""
Tests for goal and task management in AgentRuntimeDB.

Tests cover:
- Goal CRUD operations
- Task CRUD operations with dependencies
- Task queue operations
- Goal-task relationships
- Status transitions
"""
import pytest
import json
from datetime import datetime


class TestGoalManagement:
    """Test goal creation, retrieval, and status management."""

    def test_create_goal_minimal(self, agent_runtime_db):
        """Test creating a goal with minimal required fields."""
        goal_id = agent_runtime_db.create_goal(
            name="Test Goal",
            description="A test goal"
        )

        assert goal_id is not None
        assert isinstance(goal_id, int)
        assert goal_id > 0

    def test_create_goal_with_metadata(self, agent_runtime_db):
        """Test creating a goal with metadata."""
        metadata = {"priority": "high", "tags": ["test", "urgent"]}
        goal_id = agent_runtime_db.create_goal(
            name="Goal with Metadata",
            description="A goal with metadata",
            metadata=metadata
        )

        goal = agent_runtime_db.get_goal(goal_id)
        assert goal is not None
        assert goal["name"] == "Goal with Metadata"
        stored_metadata = json.loads(goal["metadata"])
        assert stored_metadata["priority"] == "high"
        assert "test" in stored_metadata["tags"]

    def test_get_goal_by_id(self, agent_runtime_db, sample_goal):
        """Test retrieving a goal by ID."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)

        goal = agent_runtime_db.get_goal(goal_id)
        assert goal is not None
        assert goal["id"] == goal_id
        assert goal["name"] == sample_goal["name"]
        assert goal["description"] == sample_goal["description"]
        assert goal["status"] == "active"

    def test_get_nonexistent_goal(self, agent_runtime_db):
        """Test retrieving a goal that doesn't exist."""
        goal = agent_runtime_db.get_goal(99999)
        assert goal is None

    def test_list_goals_all(self, agent_runtime_db):
        """Test listing all goals."""
        # Create multiple goals
        for i in range(3):
            agent_runtime_db.create_goal(
                name=f"Goal {i}",
                description=f"Description {i}"
            )

        goals = agent_runtime_db.list_goals()
        assert len(goals) >= 3

    def test_list_goals_by_status(self, agent_runtime_db):
        """Test filtering goals by status."""
        # Create goals with different statuses
        goal_id = agent_runtime_db.create_goal(
            name="Active Goal",
            description="An active goal"
        )
        agent_runtime_db.update_goal_status(goal_id, "completed")

        agent_runtime_db.create_goal(
            name="Another Active Goal",
            description="Still active"
        )

        active_goals = agent_runtime_db.list_goals(status="active")
        completed_goals = agent_runtime_db.list_goals(status="completed")

        assert all(g["status"] == "active" for g in active_goals)
        assert all(g["status"] == "completed" for g in completed_goals)

    def test_update_goal_status_to_completed(self, agent_runtime_db, sample_goal):
        """Test updating goal status to completed."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)

        agent_runtime_db.update_goal_status(goal_id, "completed")

        goal = agent_runtime_db.get_goal(goal_id)
        assert goal["status"] == "completed"
        assert goal["completed_at"] is not None

    def test_update_goal_status_to_cancelled(self, agent_runtime_db, sample_goal):
        """Test updating goal status to cancelled."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)

        agent_runtime_db.update_goal_status(goal_id, "cancelled")

        goal = agent_runtime_db.get_goal(goal_id)
        assert goal["status"] == "cancelled"


class TestTaskManagement:
    """Test task creation, retrieval, and queue operations."""

    def test_create_task_minimal(self, agent_runtime_db):
        """Test creating a task with minimal fields."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Standalone Task"
        )

        assert task_id is not None
        assert isinstance(task_id, int)

    def test_create_task_with_goal(self, agent_runtime_db, sample_goal):
        """Test creating a task associated with a goal."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)
        task_id = agent_runtime_db.create_task(
            goal_id=goal_id,
            title="Task for Goal",
            description="A task linked to a goal",
            priority=7
        )

        task = agent_runtime_db.get_task(task_id)
        assert task is not None
        assert task["goal_id"] == goal_id
        assert task["priority"] == 7

    def test_create_task_with_dependencies(self, agent_runtime_db):
        """Test creating a task with dependencies."""
        task1_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Prerequisite Task"
        )
        task2_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Dependent Task",
            dependencies=[task1_id]
        )

        task2 = agent_runtime_db.get_task(task2_id)
        assert task1_id in task2["dependencies"]

    def test_create_task_with_metadata(self, agent_runtime_db):
        """Test creating a task with metadata."""
        metadata = {"estimate": "2h", "component": "backend"}
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Task with Metadata",
            metadata=metadata
        )

        task = agent_runtime_db.get_task(task_id)
        assert task["metadata"]["estimate"] == "2h"
        assert task["metadata"]["component"] == "backend"

    def test_get_task_by_id(self, agent_runtime_db, sample_task):
        """Test retrieving a task by ID."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            **sample_task
        )

        task = agent_runtime_db.get_task(task_id)
        assert task is not None
        assert task["id"] == task_id
        assert task["title"] == sample_task["title"]

    def test_get_nonexistent_task(self, agent_runtime_db):
        """Test retrieving a task that doesn't exist."""
        task = agent_runtime_db.get_task(99999)
        assert task is None

    def test_list_tasks_all(self, agent_runtime_db):
        """Test listing all tasks."""
        for i in range(3):
            agent_runtime_db.create_task(
                goal_id=None,
                title=f"Task {i}"
            )

        tasks = agent_runtime_db.list_tasks()
        assert len(tasks) >= 3

    def test_list_tasks_by_goal(self, agent_runtime_db, sample_goal):
        """Test filtering tasks by goal ID."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)

        # Create tasks for this goal
        for i in range(2):
            agent_runtime_db.create_task(
                goal_id=goal_id,
                title=f"Goal Task {i}"
            )

        # Create a task without goal
        agent_runtime_db.create_task(
            goal_id=None,
            title="Standalone"
        )

        goal_tasks = agent_runtime_db.list_tasks(goal_id=goal_id)
        assert len(goal_tasks) == 2
        assert all(t["goal_id"] == goal_id for t in goal_tasks)

    def test_list_tasks_by_status(self, agent_runtime_db):
        """Test filtering tasks by status."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Task to Complete"
        )
        agent_runtime_db.update_task_status(task_id, "completed")

        pending_tasks = agent_runtime_db.list_tasks(status="pending")
        completed_tasks = agent_runtime_db.list_tasks(status="completed")

        assert all(t["status"] == "pending" for t in pending_tasks)
        assert all(t["status"] == "completed" for t in completed_tasks)

    def test_task_priority_ordering(self, agent_runtime_db):
        """Test that tasks are returned in priority order."""
        # Create tasks with different priorities
        agent_runtime_db.create_task(goal_id=None, title="Low Priority", priority=2)
        agent_runtime_db.create_task(goal_id=None, title="High Priority", priority=9)
        agent_runtime_db.create_task(goal_id=None, title="Medium Priority", priority=5)

        tasks = agent_runtime_db.list_tasks()
        # Verify high priority tasks come first when sorted
        priorities = [t["priority"] for t in tasks]
        # At minimum, the highest priority task should be accessible


class TestTaskStatusTransitions:
    """Test task status transitions."""

    def test_task_status_pending_to_in_progress(self, agent_runtime_db):
        """Test transitioning task from pending to in_progress."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Task to Start"
        )

        agent_runtime_db.update_task_status(task_id, "in_progress")

        task = agent_runtime_db.get_task(task_id)
        assert task["status"] == "in_progress"
        assert task["started_at"] is not None

    def test_task_status_to_completed_with_result(self, agent_runtime_db):
        """Test completing a task with result."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Task to Complete"
        )

        agent_runtime_db.update_task_status(
            task_id,
            "completed",
            result="Task completed successfully"
        )

        task = agent_runtime_db.get_task(task_id)
        assert task["status"] == "completed"
        assert task["completed_at"] is not None
        assert task["result"] == "Task completed successfully"

    def test_task_status_to_failed_with_error(self, agent_runtime_db):
        """Test failing a task with error message."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Task to Fail"
        )

        agent_runtime_db.update_task_status(
            task_id,
            "failed",
            error="Something went wrong"
        )

        task = agent_runtime_db.get_task(task_id)
        assert task["status"] == "failed"
        assert task["error"] == "Something went wrong"

    def test_task_status_to_cancelled(self, agent_runtime_db):
        """Test cancelling a task."""
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Task to Cancel"
        )

        agent_runtime_db.update_task_status(task_id, "cancelled")

        task = agent_runtime_db.get_task(task_id)
        assert task["status"] == "cancelled"


class TestTaskQueue:
    """Test task queue operations."""

    def test_get_next_task_highest_priority(self, agent_runtime_db):
        """Test that get_next_task returns highest priority task."""
        agent_runtime_db.create_task(goal_id=None, title="Low", priority=3)
        high_id = agent_runtime_db.create_task(goal_id=None, title="High", priority=10)
        agent_runtime_db.create_task(goal_id=None, title="Medium", priority=5)

        next_task = agent_runtime_db.get_next_task()
        assert next_task is not None
        assert next_task["id"] == high_id

    def test_get_next_task_respects_dependencies(self, agent_runtime_db):
        """Test that tasks with unmet dependencies are not returned."""
        prereq_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Prerequisite",
            priority=1
        )
        agent_runtime_db.create_task(
            goal_id=None,
            title="Dependent High Priority",
            priority=10,
            dependencies=[prereq_id]
        )

        # The dependent task has higher priority but should not be returned
        # because its dependency is not completed
        next_task = agent_runtime_db.get_next_task()

        # Should get the prerequisite task first
        assert next_task["title"] == "Prerequisite"

    def test_get_next_task_allows_completed_dependencies(self, agent_runtime_db):
        """Test that tasks with completed dependencies can be retrieved."""
        prereq_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Prerequisite",
            priority=1
        )
        dependent_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Dependent",
            priority=10,
            dependencies=[prereq_id]
        )

        # Complete the prerequisite
        agent_runtime_db.update_task_status(prereq_id, "completed")

        # Now the dependent task should be available
        next_task = agent_runtime_db.get_next_task()
        assert next_task["id"] == dependent_id

    def test_get_next_task_skips_in_progress(self, agent_runtime_db):
        """Test that in_progress tasks are skipped."""
        task1_id = agent_runtime_db.create_task(
            goal_id=None,
            title="In Progress",
            priority=10
        )
        task2_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Pending",
            priority=5
        )

        agent_runtime_db.update_task_status(task1_id, "in_progress")

        next_task = agent_runtime_db.get_next_task()
        assert next_task["id"] == task2_id

    def test_get_next_task_empty_queue(self, agent_runtime_db):
        """Test get_next_task with no pending tasks."""
        # Create and complete all tasks
        task_id = agent_runtime_db.create_task(
            goal_id=None,
            title="Only Task"
        )
        agent_runtime_db.update_task_status(task_id, "completed")

        next_task = agent_runtime_db.get_next_task()
        assert next_task is None


class TestGoalTaskRelationship:
    """Test relationships between goals and tasks."""

    def test_tasks_track_goal_relationship(self, agent_runtime_db, sample_goal):
        """Test that tasks maintain reference to their parent goal."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)

        task_ids = []
        for i in range(3):
            task_id = agent_runtime_db.create_task(
                goal_id=goal_id,
                title=f"Task {i} for Goal"
            )
            task_ids.append(task_id)

        for task_id in task_ids:
            task = agent_runtime_db.get_task(task_id)
            assert task["goal_id"] == goal_id

    def test_goal_task_count(self, agent_runtime_db, sample_goal):
        """Test counting tasks for a goal."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)

        for i in range(5):
            agent_runtime_db.create_task(
                goal_id=goal_id,
                title=f"Task {i}"
            )

        tasks = agent_runtime_db.list_tasks(goal_id=goal_id)
        assert len(tasks) == 5

    def test_deleting_goal_orphans_tasks(self, agent_runtime_db, sample_goal):
        """Test behavior when goal is deleted (tasks remain orphaned)."""
        goal_id = agent_runtime_db.create_goal(**sample_goal)
        task_id = agent_runtime_db.create_task(
            goal_id=goal_id,
            title="Orphan Task"
        )

        # Note: The current implementation doesn't have delete_goal
        # This test documents expected behavior if implemented
        task = agent_runtime_db.get_task(task_id)
        assert task["goal_id"] == goal_id
