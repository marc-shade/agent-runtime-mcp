"""
Circuit Breaker (Tiny Dancer Pattern) - God Agent Phase 5.1

Implements fault tolerance for multi-agent relay pipelines with:
- Three states: CLOSED (normal) -> OPEN (suspended) -> HALF_OPEN (trial)
- Per-agent failure tracking with sliding window
- Automatic recovery via half-open trial state
- Graceful degradation to fallback agents
- Integration with relay orchestrator

From the God Agent paper:
"The Tiny Dancer pattern prevents cascading failures by suspending
 unreliable agents and routing to fallbacks, with automatic
 recovery testing after cooldown periods."

Author: Claude Code (God Agent Integration)
Created: 2025-01-03
"""

import sqlite3
import json
import logging
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from pathlib import Path
import threading


logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"      # Normal operation - all requests pass through
    OPEN = "open"          # Suspended - requests blocked, using fallback
    HALF_OPEN = "half_open"  # Trial - allowing limited requests to test recovery


class FailureType(Enum):
    """Types of failures that can trigger the circuit breaker."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    QUALITY_FAILURE = "quality_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    RATE_LIMITED = "rate_limited"
    INVALID_OUTPUT = "invalid_output"


class RecoveryAction(Enum):
    """Actions taken during recovery."""
    RETRY_SAME = "retry_same"      # Retry with same agent
    USE_FALLBACK = "use_fallback"  # Route to fallback agent
    ESCALATE = "escalate"          # Escalate to orchestrator
    ABORT = "abort"                # Abort the operation


# Default configuration
DEFAULT_CONFIG = {
    "failure_threshold": 5,       # Failures before tripping
    "window_seconds": 60,         # Sliding window for counting failures
    "cooldown_seconds": 300,      # Time before half-open trial (5 min)
    "half_open_max_calls": 3,     # Max calls in half-open state
    "success_threshold": 2,       # Successes to close from half-open
    "fallback_agent": "generalist",
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FailureRecord:
    """Record of a single failure event."""
    failure_id: str
    agent_id: str
    failure_type: FailureType
    error_message: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "agent_id": self.agent_id,
            "failure_type": self.failure_type.value,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


@dataclass
class CircuitBreakerState:
    """Complete state of a circuit breaker for one agent."""
    agent_id: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    half_open_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    trip_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "half_open_calls": self.half_open_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "trip_count": self.trip_count,
            "created_at": self.created_at.isoformat(),
            "config": self.config
        }


@dataclass
class ExecutionResult:
    """Result of executing with circuit breaker protection."""
    success: bool
    result: Any
    agent_used: str
    original_agent: str
    used_fallback: bool
    circuit_state: CircuitState
    execution_time_ms: int
    error: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

class CircuitBreaker:
    """
    Circuit Breaker for a single agent.

    State Machine:
        CLOSED --[failures >= threshold]--> OPEN
        OPEN --[cooldown elapsed]--> HALF_OPEN
        HALF_OPEN --[success >= threshold]--> CLOSED
        HALF_OPEN --[any failure]--> OPEN

    God Agent Integration:
        Used by RelayOrchestrator to protect relay race execution.
        Each agent in the pipeline has its own circuit breaker.
    """

    def __init__(
        self,
        agent_id: str,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        cooldown_seconds: int = 300,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        fallback_agent: str = "generalist"
    ):
        self.agent_id = agent_id
        self.state = CircuitBreakerState(
            agent_id=agent_id,
            config={
                "failure_threshold": failure_threshold,
                "window_seconds": window_seconds,
                "cooldown_seconds": cooldown_seconds,
                "half_open_max_calls": half_open_max_calls,
                "success_threshold": success_threshold,
                "fallback_agent": fallback_agent
            }
        )
        self.failures: List[FailureRecord] = []
        self._lock = threading.Lock()

    @property
    def current_state(self) -> CircuitState:
        """Get current state, checking for automatic transitions."""
        with self._lock:
            self._check_state_transitions()
            return self.state.state

    def _check_state_transitions(self) -> None:
        """Check and perform automatic state transitions."""
        now = datetime.now()

        if self.state.state == CircuitState.OPEN:
            # Check if cooldown period has elapsed
            if self.state.cooldown_until and now >= self.state.cooldown_until:
                self._transition_to_half_open()

    def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN for recovery trial."""
        self.state.state = CircuitState.HALF_OPEN
        self.state.half_open_calls = 0
        self.state.success_count = 0
        logger.info(f"Circuit breaker {self.agent_id}: OPEN -> HALF_OPEN (recovery trial)")

    def can_execute(self) -> bool:
        """Check if a request can be executed through this agent."""
        state = self.current_state

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        if state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            max_calls = self.state.config["half_open_max_calls"]
            return self.state.half_open_calls < max_calls

        return False

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            now = datetime.now()
            self.state.last_success_time = now
            self.state.total_successes += 1

            if self.state.state == CircuitState.HALF_OPEN:
                self.state.success_count += 1
                self.state.half_open_calls += 1

                # Check if we should close the circuit
                if self.state.success_count >= self.state.config["success_threshold"]:
                    self._close_circuit()
            elif self.state.state == CircuitState.CLOSED:
                # Reset failure count on success (sliding window effect)
                self._clean_old_failures()

    def record_failure(
        self,
        failure_type: FailureType,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a failure and potentially trip the circuit."""
        with self._lock:
            now = datetime.now()

            # Create failure record
            failure = FailureRecord(
                failure_id=f"{self.agent_id}-{now.timestamp()}",
                agent_id=self.agent_id,
                failure_type=failure_type,
                error_message=error_message,
                timestamp=now,
                context=context or {}
            )
            self.failures.append(failure)

            self.state.last_failure_time = now
            self.state.total_failures += 1

            if self.state.state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately trips back to open
                self._trip_circuit("failure in half-open state")
            elif self.state.state == CircuitState.CLOSED:
                self._clean_old_failures()
                self.state.failure_count = len(self.failures)

                # Check if we should trip
                if self.state.failure_count >= self.state.config["failure_threshold"]:
                    self._trip_circuit(f"{self.state.failure_count} failures in window")

    def _trip_circuit(self, reason: str) -> None:
        """Trip the circuit to OPEN state."""
        self.state.state = CircuitState.OPEN
        cooldown = self.state.config["cooldown_seconds"]
        self.state.cooldown_until = datetime.now() + timedelta(seconds=cooldown)
        self.state.trip_count += 1

        logger.warning(
            f"Circuit breaker TRIPPED for {self.agent_id}: {reason}. "
            f"Cooldown until {self.state.cooldown_until.isoformat()}"
        )

    def _close_circuit(self) -> None:
        """Close the circuit after successful recovery."""
        self.state.state = CircuitState.CLOSED
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.half_open_calls = 0
        self.state.cooldown_until = None
        self.failures.clear()

        logger.info(f"Circuit breaker {self.agent_id}: HALF_OPEN -> CLOSED (recovered)")

    def _clean_old_failures(self) -> None:
        """Remove failures outside the sliding window."""
        window = self.state.config["window_seconds"]
        cutoff = datetime.now() - timedelta(seconds=window)
        self.failures = [f for f in self.failures if f.timestamp >= cutoff]
        self.state.failure_count = len(self.failures)

    def force_open(self, reason: str = "manual intervention") -> None:
        """Manually open the circuit."""
        with self._lock:
            self._trip_circuit(reason)

    def force_close(self) -> None:
        """Manually close the circuit."""
        with self._lock:
            self._close_circuit()

    def get_status(self) -> Dict[str, Any]:
        """Get complete circuit breaker status."""
        return {
            **self.state.to_dict(),
            "can_execute": self.can_execute(),
            "recent_failures": [f.to_dict() for f in self.failures[-10:]],
            "time_until_trial": self._time_until_trial()
        }

    def _time_until_trial(self) -> Optional[int]:
        """Seconds until half-open trial (if in OPEN state)."""
        if self.state.state != CircuitState.OPEN:
            return None
        if not self.state.cooldown_until:
            return None

        remaining = (self.state.cooldown_until - datetime.now()).total_seconds()
        return max(0, int(remaining))


# ============================================================================
# Circuit Breaker Registry
# ============================================================================

class CircuitBreakerRegistry:
    """
    Registry managing circuit breakers for all agents.

    Provides centralized management, persistence, and monitoring
    of all circuit breakers in the system.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.db_path = db_path
        self._lock = threading.Lock()

        if db_path:
            self._init_database()
            self._load_persisted_state()

    def _init_database(self) -> None:
        """Initialize circuit breaker persistence tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Circuit breaker state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breakers (
                agent_id TEXT PRIMARY KEY,
                state TEXT NOT NULL DEFAULT 'closed',
                failure_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                total_failures INTEGER DEFAULT 0,
                total_successes INTEGER DEFAULT 0,
                trip_count INTEGER DEFAULT 0,
                last_failure_time TEXT,
                last_success_time TEXT,
                cooldown_until TEXT,
                config TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Failure history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_failures (
                failure_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                error_message TEXT,
                context TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES circuit_breakers(agent_id)
            )
        """)

        # Circuit events table (for audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS circuit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT,
                reason TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (agent_id) REFERENCES circuit_breakers(agent_id)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_failures_agent
            ON circuit_failures(agent_id, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_agent
            ON circuit_events(agent_id, timestamp)
        """)

        conn.commit()
        conn.close()
        logger.info(f"Circuit breaker database initialized at {self.db_path}")

    def _load_persisted_state(self) -> None:
        """Load persisted circuit breaker states from database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM circuit_breakers")
        rows = cursor.fetchall()

        for row in rows:
            agent_id = row[0]
            config = json.loads(row[10]) if row[10] else DEFAULT_CONFIG.copy()

            breaker = CircuitBreaker(
                agent_id=agent_id,
                **{k: v for k, v in config.items() if k != "fallback_agent"},
                fallback_agent=config.get("fallback_agent", "generalist")
            )

            # Restore state
            breaker.state.state = CircuitState(row[1])
            breaker.state.failure_count = row[2]
            breaker.state.success_count = row[3]
            breaker.state.total_failures = row[4]
            breaker.state.total_successes = row[5]
            breaker.state.trip_count = row[6]

            if row[7]:
                breaker.state.last_failure_time = datetime.fromisoformat(row[7])
            if row[8]:
                breaker.state.last_success_time = datetime.fromisoformat(row[8])
            if row[9]:
                breaker.state.cooldown_until = datetime.fromisoformat(row[9])

            self.breakers[agent_id] = breaker

        conn.close()
        logger.info(f"Loaded {len(self.breakers)} circuit breaker states from database")

    def get_or_create(
        self,
        agent_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker for an agent."""
        with self._lock:
            if agent_id not in self.breakers:
                cfg = {**DEFAULT_CONFIG, **(config or {})}
                self.breakers[agent_id] = CircuitBreaker(
                    agent_id=agent_id,
                    **{k: v for k, v in cfg.items() if k != "fallback_agent"},
                    fallback_agent=cfg.get("fallback_agent", "generalist")
                )
                self._persist_breaker(agent_id)

            return self.breakers[agent_id]

    def _persist_breaker(self, agent_id: str) -> None:
        """Persist circuit breaker state to database."""
        if not self.db_path:
            return

        breaker = self.breakers.get(agent_id)
        if not breaker:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        state = breaker.state
        cursor.execute("""
            INSERT OR REPLACE INTO circuit_breakers (
                agent_id, state, failure_count, success_count,
                total_failures, total_successes, trip_count,
                last_failure_time, last_success_time, cooldown_until,
                config, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.agent_id,
            state.state.value,
            state.failure_count,
            state.success_count,
            state.total_failures,
            state.total_successes,
            state.trip_count,
            state.last_failure_time.isoformat() if state.last_failure_time else None,
            state.last_success_time.isoformat() if state.last_success_time else None,
            state.cooldown_until.isoformat() if state.cooldown_until else None,
            json.dumps(state.config),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def record_event(
        self,
        agent_id: str,
        event_type: str,
        from_state: Optional[str] = None,
        to_state: Optional[str] = None,
        reason: Optional[str] = None
    ) -> None:
        """Record a circuit breaker event for audit trail."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO circuit_events (agent_id, event_type, from_state, to_state, reason)
            VALUES (?, ?, ?, ?, ?)
        """, (agent_id, event_type, from_state, to_state, reason))

        conn.commit()
        conn.close()

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            agent_id: breaker.get_status()
            for agent_id, breaker in self.breakers.items()
        }

    def get_open_circuits(self) -> List[str]:
        """Get list of agents with open circuits."""
        return [
            agent_id
            for agent_id, breaker in self.breakers.items()
            if breaker.current_state == CircuitState.OPEN
        ]

    def get_half_open_circuits(self) -> List[str]:
        """Get list of agents in half-open state."""
        return [
            agent_id
            for agent_id, breaker in self.breakers.items()
            if breaker.current_state == CircuitState.HALF_OPEN
        ]

    def reset_all(self) -> int:
        """Reset all circuit breakers to CLOSED state."""
        count = 0
        for breaker in self.breakers.values():
            if breaker.state.state != CircuitState.CLOSED:
                breaker.force_close()
                count += 1
        return count


# ============================================================================
# Execution with Fault Tolerance
# ============================================================================

class FaultTolerantExecutor:
    """
    Executor that wraps agent calls with circuit breaker protection.

    Provides:
    - Automatic fallback routing when agents are unavailable
    - Graceful degradation with quality awareness
    - Recovery testing for suspended agents
    - Integration with relay orchestrator
    """

    def __init__(self, registry: CircuitBreakerRegistry):
        self.registry = registry
        self.default_fallback = "generalist"

    async def execute(
        self,
        agent_id: str,
        task: Callable[[], Awaitable[Any]],
        timeout_seconds: int = 60,
        fallback_task: Optional[Callable[[], Awaitable[Any]]] = None,
        quality_threshold: float = 0.3
    ) -> ExecutionResult:
        """
        Execute a task with circuit breaker protection.

        Args:
            agent_id: The agent to execute with
            task: Async callable representing the task
            timeout_seconds: Maximum execution time
            fallback_task: Optional fallback task
            quality_threshold: Minimum quality to consider success

        Returns:
            ExecutionResult with success status and details
        """
        import time

        start_time = time.time()
        breaker = self.registry.get_or_create(agent_id)

        # Check if we can execute with primary agent
        if not breaker.can_execute():
            # Route to fallback
            if fallback_task:
                return await self._execute_fallback(
                    agent_id, fallback_task, start_time, breaker
                )
            else:
                return ExecutionResult(
                    success=False,
                    result=None,
                    agent_used=agent_id,
                    original_agent=agent_id,
                    used_fallback=False,
                    circuit_state=breaker.current_state,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    error=f"Circuit breaker OPEN for {agent_id}",
                    recovery_action=RecoveryAction.ABORT
                )

        # Execute with primary agent
        try:
            result = await asyncio.wait_for(
                task(),
                timeout=timeout_seconds
            )

            # Record success
            breaker.record_success()
            self.registry._persist_breaker(agent_id)

            return ExecutionResult(
                success=True,
                result=result,
                agent_used=agent_id,
                original_agent=agent_id,
                used_fallback=False,
                circuit_state=breaker.current_state,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

        except asyncio.TimeoutError:
            breaker.record_failure(
                FailureType.TIMEOUT,
                f"Timeout after {timeout_seconds}s"
            )
            self.registry._persist_breaker(agent_id)

            # Try fallback
            if fallback_task and breaker.current_state == CircuitState.OPEN:
                return await self._execute_fallback(
                    agent_id, fallback_task, start_time, breaker
                )

            return ExecutionResult(
                success=False,
                result=None,
                agent_used=agent_id,
                original_agent=agent_id,
                used_fallback=False,
                circuit_state=breaker.current_state,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=f"Timeout after {timeout_seconds}s",
                recovery_action=RecoveryAction.RETRY_SAME if breaker.can_execute() else RecoveryAction.USE_FALLBACK
            )

        except Exception as e:
            breaker.record_failure(
                FailureType.EXCEPTION,
                str(e),
                {"exception_type": type(e).__name__}
            )
            self.registry._persist_breaker(agent_id)

            # Try fallback if circuit tripped
            if fallback_task and breaker.current_state == CircuitState.OPEN:
                return await self._execute_fallback(
                    agent_id, fallback_task, start_time, breaker
                )

            return ExecutionResult(
                success=False,
                result=None,
                agent_used=agent_id,
                original_agent=agent_id,
                used_fallback=False,
                circuit_state=breaker.current_state,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=str(e),
                recovery_action=RecoveryAction.RETRY_SAME if breaker.can_execute() else RecoveryAction.USE_FALLBACK
            )

    async def _execute_fallback(
        self,
        original_agent: str,
        fallback_task: Callable[[], Awaitable[Any]],
        start_time: float,
        original_breaker: CircuitBreaker
    ) -> ExecutionResult:
        """Execute fallback task when primary agent is unavailable."""
        import time

        fallback_agent = original_breaker.state.config.get("fallback_agent", self.default_fallback)

        try:
            result = await fallback_task()

            return ExecutionResult(
                success=True,
                result=result,
                agent_used=fallback_agent,
                original_agent=original_agent,
                used_fallback=True,
                circuit_state=original_breaker.current_state,
                execution_time_ms=int((time.time() - start_time) * 1000),
                recovery_action=RecoveryAction.USE_FALLBACK
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                agent_used=fallback_agent,
                original_agent=original_agent,
                used_fallback=True,
                circuit_state=original_breaker.current_state,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error=f"Fallback also failed: {e}",
                recovery_action=RecoveryAction.ESCALATE
            )


# ============================================================================
# MCP Tool Registration
# ============================================================================

def register_circuit_breaker_tools(app, db_path: str):
    """
    Register circuit breaker MCP tools for external access.

    Tools:
    - circuit_breaker_status: Get status of a circuit breaker
    - circuit_breaker_list: List all circuit breakers
    - circuit_breaker_trip: Manually trip a circuit
    - circuit_breaker_reset: Reset a circuit breaker
    - circuit_breaker_execute: Execute with circuit breaker protection
    - circuit_breaker_events: Get circuit breaker event history
    """

    registry = CircuitBreakerRegistry(db_path)
    executor = FaultTolerantExecutor(registry)

    @app.tool()
    async def circuit_breaker_status(agent_id: str) -> str:
        """
        Get the status of a circuit breaker for an agent.

        Args:
            agent_id: The agent identifier

        Returns:
            JSON status including state, failure count, and configuration
        """
        breaker = registry.get_or_create(agent_id)
        return json.dumps(breaker.get_status(), indent=2, default=str)

    @app.tool()
    async def circuit_breaker_list() -> str:
        """
        List all circuit breakers and their current states.

        Returns:
            JSON object with all circuit breaker statuses
        """
        status = registry.get_all_status()

        summary = {
            "total_breakers": len(status),
            "open_circuits": registry.get_open_circuits(),
            "half_open_circuits": registry.get_half_open_circuits(),
            "breakers": status
        }

        return json.dumps(summary, indent=2, default=str)

    @app.tool()
    async def circuit_breaker_trip(agent_id: str, reason: str = "manual intervention") -> str:
        """
        Manually trip a circuit breaker to OPEN state.

        Args:
            agent_id: The agent to trip
            reason: Reason for manual trip

        Returns:
            Confirmation message
        """
        breaker = registry.get_or_create(agent_id)
        old_state = breaker.current_state.value
        breaker.force_open(reason)
        registry._persist_breaker(agent_id)

        registry.record_event(
            agent_id, "manual_trip",
            from_state=old_state,
            to_state="open",
            reason=reason
        )

        return json.dumps({
            "success": True,
            "agent_id": agent_id,
            "action": "trip",
            "old_state": old_state,
            "new_state": "open",
            "reason": reason,
            "cooldown_until": breaker.state.cooldown_until.isoformat()
        }, indent=2)

    @app.tool()
    async def circuit_breaker_reset(agent_id: str) -> str:
        """
        Reset a circuit breaker to CLOSED state.

        Args:
            agent_id: The agent to reset

        Returns:
            Confirmation message
        """
        breaker = registry.get_or_create(agent_id)
        old_state = breaker.current_state.value
        breaker.force_close()
        registry._persist_breaker(agent_id)

        registry.record_event(
            agent_id, "manual_reset",
            from_state=old_state,
            to_state="closed",
            reason="manual reset"
        )

        return json.dumps({
            "success": True,
            "agent_id": agent_id,
            "action": "reset",
            "old_state": old_state,
            "new_state": "closed"
        }, indent=2)

    @app.tool()
    async def circuit_breaker_configure(
        agent_id: str,
        failure_threshold: int = 5,
        window_seconds: int = 60,
        cooldown_seconds: int = 300,
        fallback_agent: str = "generalist"
    ) -> str:
        """
        Configure a circuit breaker for an agent.

        Args:
            agent_id: The agent to configure
            failure_threshold: Number of failures before tripping (default: 5)
            window_seconds: Sliding window for counting failures (default: 60)
            cooldown_seconds: Cooldown before recovery trial (default: 300)
            fallback_agent: Agent to use when circuit is open (default: generalist)

        Returns:
            Configuration confirmation
        """
        config = {
            "failure_threshold": failure_threshold,
            "window_seconds": window_seconds,
            "cooldown_seconds": cooldown_seconds,
            "fallback_agent": fallback_agent,
            "half_open_max_calls": 3,
            "success_threshold": 2
        }

        breaker = registry.get_or_create(agent_id, config)
        breaker.state.config.update(config)
        registry._persist_breaker(agent_id)

        return json.dumps({
            "success": True,
            "agent_id": agent_id,
            "config": config
        }, indent=2)

    @app.tool()
    async def circuit_breaker_events(
        agent_id: str,
        limit: int = 20
    ) -> str:
        """
        Get circuit breaker event history for an agent.

        Args:
            agent_id: The agent to get events for
            limit: Maximum number of events (default: 20)

        Returns:
            JSON array of events
        """
        if not registry.db_path:
            return json.dumps({"error": "No database configured"})

        conn = sqlite3.connect(registry.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT event_type, from_state, to_state, reason, timestamp
            FROM circuit_events
            WHERE agent_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (agent_id, limit))

        events = [
            {
                "event_type": row[0],
                "from_state": row[1],
                "to_state": row[2],
                "reason": row[3],
                "timestamp": row[4]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        return json.dumps({
            "agent_id": agent_id,
            "event_count": len(events),
            "events": events
        }, indent=2)

    @app.tool()
    async def circuit_breaker_record_failure(
        agent_id: str,
        failure_type: str,
        error_message: str
    ) -> str:
        """
        Record a failure for a circuit breaker.

        Args:
            agent_id: The agent that failed
            failure_type: Type of failure (timeout, exception, quality_failure, etc.)
            error_message: Description of the failure

        Returns:
            Updated circuit breaker status
        """
        breaker = registry.get_or_create(agent_id)

        try:
            ftype = FailureType(failure_type)
        except ValueError:
            ftype = FailureType.EXCEPTION

        old_state = breaker.current_state.value
        breaker.record_failure(ftype, error_message)
        registry._persist_breaker(agent_id)

        new_state = breaker.current_state.value

        if old_state != new_state:
            registry.record_event(
                agent_id, "failure_trip",
                from_state=old_state,
                to_state=new_state,
                reason=error_message
            )

        return json.dumps({
            "success": True,
            "agent_id": agent_id,
            "failure_recorded": True,
            "old_state": old_state,
            "new_state": new_state,
            "tripped": old_state != new_state,
            "status": breaker.get_status()
        }, indent=2, default=str)

    @app.tool()
    async def circuit_breaker_record_success(agent_id: str) -> str:
        """
        Record a success for a circuit breaker.

        Args:
            agent_id: The agent that succeeded

        Returns:
            Updated circuit breaker status
        """
        breaker = registry.get_or_create(agent_id)

        old_state = breaker.current_state.value
        breaker.record_success()
        registry._persist_breaker(agent_id)

        new_state = breaker.current_state.value

        if old_state != new_state:
            registry.record_event(
                agent_id, "recovery",
                from_state=old_state,
                to_state=new_state,
                reason="successful recovery trial"
            )

        return json.dumps({
            "success": True,
            "agent_id": agent_id,
            "success_recorded": True,
            "old_state": old_state,
            "new_state": new_state,
            "recovered": old_state == "half_open" and new_state == "closed",
            "status": breaker.get_status()
        }, indent=2, default=str)

    logger.info("Circuit breaker tools registered (God Agent Phase 5: Fault Tolerance)")
    return registry, executor


# ============================================================================
# Integration Helper Functions
# ============================================================================

async def execute_with_circuit_breaker(
    registry: CircuitBreakerRegistry,
    agent_id: str,
    task: Callable[[], Awaitable[Any]],
    fallback: Optional[Callable[[], Awaitable[Any]]] = None,
    timeout: int = 60
) -> ExecutionResult:
    """
    Convenience function to execute a task with circuit breaker protection.

    Example:
        result = await execute_with_circuit_breaker(
            registry,
            agent_id="code_reviewer",
            task=lambda: review_code(code),
            fallback=lambda: simple_lint(code),
            timeout=30
        )
    """
    executor = FaultTolerantExecutor(registry)
    return await executor.execute(
        agent_id=agent_id,
        task=task,
        fallback_task=fallback,
        timeout_seconds=timeout
    )


def get_healthy_agents(registry: CircuitBreakerRegistry) -> List[str]:
    """Get list of agents with closed circuit breakers."""
    return [
        agent_id
        for agent_id, breaker in registry.breakers.items()
        if breaker.current_state == CircuitState.CLOSED
    ]


def get_degraded_agents(registry: CircuitBreakerRegistry) -> List[str]:
    """Get list of agents with non-closed circuit breakers."""
    return [
        agent_id
        for agent_id, breaker in registry.breakers.items()
        if breaker.current_state != CircuitState.CLOSED
    ]


# ============================================================================
# Standalone Test
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_circuit_breaker():
        """Test circuit breaker functionality."""
        print("Testing Circuit Breaker (Tiny Dancer Pattern)")
        print("=" * 60)

        # Create registry with in-memory database
        import tempfile
        db_path = tempfile.mktemp(suffix=".db")
        registry = CircuitBreakerRegistry(db_path)

        # Test 1: Basic circuit breaker
        print("\n1. Testing basic circuit breaker...")
        breaker = registry.get_or_create("test_agent", {
            "failure_threshold": 3,
            "window_seconds": 60,
            "cooldown_seconds": 5
        })

        assert breaker.current_state == CircuitState.CLOSED
        assert breaker.can_execute()
        print("   Initial state: CLOSED")

        # Test 2: Record failures
        print("\n2. Recording failures...")
        for i in range(3):
            breaker.record_failure(FailureType.EXCEPTION, f"Test failure {i+1}")
            print(f"   Failure {i+1}: state = {breaker.current_state.value}")

        assert breaker.current_state == CircuitState.OPEN
        assert not breaker.can_execute()
        print("   Circuit tripped to OPEN")

        # Test 3: Wait for cooldown
        print("\n3. Waiting for cooldown (5 seconds)...")
        await asyncio.sleep(6)

        assert breaker.current_state == CircuitState.HALF_OPEN
        print("   Transitioned to HALF_OPEN")

        # Test 4: Successful recovery
        print("\n4. Testing recovery...")
        breaker.record_success()
        breaker.record_success()

        assert breaker.current_state == CircuitState.CLOSED
        print("   Recovered to CLOSED")

        # Test 5: Execution with fault tolerance
        print("\n5. Testing fault tolerant execution...")
        executor = FaultTolerantExecutor(registry)

        # Configure test_agent_2 with low threshold for testing
        test_breaker_2 = registry.get_or_create("test_agent_2", {
            "failure_threshold": 3,
            "window_seconds": 60,
            "cooldown_seconds": 5
        })

        # Create a failing task
        call_count = 0
        async def failing_task():
            nonlocal call_count
            call_count += 1
            raise Exception("Simulated failure")

        async def fallback_task():
            return "Fallback result"

        # Trip the breaker with failures (need 3 to trip)
        for i in range(3):
            result = await executor.execute(
                "test_agent_2",
                failing_task,
                timeout_seconds=5,
                fallback_task=fallback_task
            )
            print(f"   Call {i+1}: success={result.success}, used_fallback={result.used_fallback}, state={result.circuit_state.value}")

        # Next call should use fallback since circuit is OPEN
        result = await executor.execute(
            "test_agent_2",
            failing_task,
            timeout_seconds=5,
            fallback_task=fallback_task
        )

        assert result.used_fallback
        assert result.result == "Fallback result"
        print(f"   Fallback used: {result.used_fallback}")
        print(f"   Result: {result.result}")

        # Test 6: Status summary
        print("\n6. Status summary...")
        all_status = registry.get_all_status()
        print(f"   Total breakers: {len(all_status)}")
        print(f"   Open circuits: {registry.get_open_circuits()}")

        # Clean up
        import os
        os.unlink(db_path)

        print("\n" + "=" * 60)
        print("All circuit breaker tests passed!")
        return True

    # Run tests
    asyncio.run(test_circuit_breaker())
