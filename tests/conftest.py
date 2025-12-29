"""
Pytest fixtures for agent-runtime-mcp tests.
"""
import pytest
import tempfile
import sqlite3
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def db_connection(temp_db):
    """Create a database connection."""
    conn = sqlite3.connect(temp_db)
    yield conn
    conn.close()


@pytest.fixture
def agent_runtime_db(temp_db):
    """Create an AgentRuntimeDB instance with temp database."""
    from server import AgentRuntimeDB
    return AgentRuntimeDB(temp_db)


@pytest.fixture
def circuit_breaker_registry(temp_db):
    """Create a CircuitBreakerRegistry with temp database."""
    from circuit_breaker import CircuitBreakerRegistry
    return CircuitBreakerRegistry(str(temp_db))


@pytest.fixture
def sample_goal():
    """Sample goal data for testing."""
    return {
        "name": "Test Goal",
        "description": "A test goal for unit testing",
        "metadata": {"priority": "high", "tags": ["test"]}
    }


@pytest.fixture
def sample_task():
    """Sample task data for testing."""
    return {
        "title": "Test Task",
        "description": "A test task for unit testing",
        "priority": 5,
        "dependencies": [],
        "metadata": {"estimate": "1h"}
    }
