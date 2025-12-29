"""
Tests for Circuit Breaker (Tiny Dancer Pattern) implementation.

Tests cover:
- Circuit state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure recording and threshold triggering
- Success recording and recovery
- Cooldown period handling
- Manual trip and reset operations
- Registry management
"""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    FailureType,
    CircuitBreakerState,
    FailureRecord,
    FaultTolerantExecutor,
    ExecutionResult,
    RecoveryAction,
    DEFAULT_CONFIG
)


class TestCircuitBreakerStates:
    """Test circuit breaker state machine."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker("test_agent")
        assert breaker.current_state == CircuitState.CLOSED

    def test_can_execute_when_closed(self):
        """Should allow execution when circuit is CLOSED."""
        breaker = CircuitBreaker("test_agent")
        assert breaker.can_execute() is True

    def test_cannot_execute_when_open(self):
        """Should block execution when circuit is OPEN."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=2,
            cooldown_seconds=300
        )

        # Trip the circuit
        breaker.record_failure(FailureType.EXCEPTION, "Error 1")
        breaker.record_failure(FailureType.EXCEPTION, "Error 2")

        assert breaker.current_state == CircuitState.OPEN
        assert breaker.can_execute() is False

    def test_limited_execution_in_half_open(self):
        """Should allow limited execution when circuit is HALF_OPEN."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=2,
            cooldown_seconds=0,  # Immediate transition for testing
            half_open_max_calls=2
        )

        # Trip and wait for half-open
        breaker.record_failure(FailureType.EXCEPTION, "Error 1")
        breaker.record_failure(FailureType.EXCEPTION, "Error 2")

        # Force transition to HALF_OPEN by setting cooldown in past
        breaker.state.cooldown_until = datetime.now() - timedelta(seconds=1)

        assert breaker.current_state == CircuitState.HALF_OPEN
        assert breaker.can_execute() is True


class TestCircuitBreakerTransitions:
    """Test state transitions in circuit breaker."""

    def test_closed_to_open_on_threshold(self):
        """Circuit should trip from CLOSED to OPEN when failure threshold reached."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=3
        )

        breaker.record_failure(FailureType.EXCEPTION, "Error 1")
        assert breaker.current_state == CircuitState.CLOSED

        breaker.record_failure(FailureType.EXCEPTION, "Error 2")
        assert breaker.current_state == CircuitState.CLOSED

        breaker.record_failure(FailureType.EXCEPTION, "Error 3")
        assert breaker.current_state == CircuitState.OPEN

    def test_open_to_half_open_after_cooldown(self):
        """Circuit should transition to HALF_OPEN after cooldown."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=1,
            cooldown_seconds=300  # Non-zero to capture OPEN state
        )

        breaker.record_failure(FailureType.EXCEPTION, "Error")
        assert breaker.current_state == CircuitState.OPEN

        # Set cooldown in the past to trigger transition
        breaker.state.cooldown_until = datetime.now() - timedelta(seconds=1)

        # Accessing current_state triggers transition check
        assert breaker.current_state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """Circuit should close from HALF_OPEN after success threshold met."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=1,
            cooldown_seconds=0,
            success_threshold=2
        )

        # Trip and transition to half-open
        breaker.record_failure(FailureType.EXCEPTION, "Error")
        breaker.state.cooldown_until = datetime.now() - timedelta(seconds=1)
        _ = breaker.current_state  # Trigger transition

        assert breaker.current_state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.current_state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.current_state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Circuit should immediately reopen on any failure in HALF_OPEN."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=1,
            cooldown_seconds=300  # Non-zero to capture OPEN state
        )

        # Trip the circuit
        breaker.record_failure(FailureType.EXCEPTION, "Error 1")
        assert breaker.current_state == CircuitState.OPEN

        # Manually transition to half-open by setting cooldown in past
        breaker.state.cooldown_until = datetime.now() - timedelta(seconds=1)
        _ = breaker.current_state  # Trigger check

        assert breaker.current_state == CircuitState.HALF_OPEN

        # Single failure should trip back to OPEN
        breaker.record_failure(FailureType.EXCEPTION, "Error 2")

        # The circuit should go back to OPEN (check internal state directly since cooldown is set in past)
        assert breaker.state.state == CircuitState.OPEN


class TestFailureTracking:
    """Test failure recording and tracking."""

    def test_record_failure_increments_count(self):
        """Recording failure should increment failure count."""
        breaker = CircuitBreaker("test_agent", failure_threshold=10)

        breaker.record_failure(FailureType.TIMEOUT, "Timeout error")
        assert breaker.state.failure_count == 1
        assert breaker.state.total_failures == 1

    def test_failure_types_tracked(self):
        """Different failure types should be properly tracked."""
        breaker = CircuitBreaker("test_agent", failure_threshold=10)

        breaker.record_failure(FailureType.TIMEOUT, "Timeout")
        breaker.record_failure(FailureType.EXCEPTION, "Exception")
        breaker.record_failure(FailureType.QUALITY_FAILURE, "Low quality")
        breaker.record_failure(FailureType.RATE_LIMITED, "Rate limited")

        assert breaker.state.total_failures == 4
        assert len(breaker.failures) == 4

        failure_types = [f.failure_type for f in breaker.failures]
        assert FailureType.TIMEOUT in failure_types
        assert FailureType.EXCEPTION in failure_types
        assert FailureType.QUALITY_FAILURE in failure_types
        assert FailureType.RATE_LIMITED in failure_types

    def test_failure_sliding_window(self):
        """Failures outside window should be cleaned up."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=3,
            window_seconds=1
        )

        breaker.record_failure(FailureType.EXCEPTION, "Old error")

        # Wait for window to expire
        time.sleep(1.5)

        # New failures should not count old ones
        breaker.record_failure(FailureType.EXCEPTION, "New error 1")
        breaker.record_failure(FailureType.EXCEPTION, "New error 2")

        # Should still be CLOSED because old failure expired
        assert breaker.state.failure_count == 2
        assert breaker.current_state == CircuitState.CLOSED

    def test_failure_context_stored(self):
        """Failure context should be stored with record."""
        breaker = CircuitBreaker("test_agent")

        context = {"request_id": "123", "endpoint": "/api/test"}
        breaker.record_failure(FailureType.EXCEPTION, "Error", context=context)

        assert len(breaker.failures) == 1
        assert breaker.failures[0].context["request_id"] == "123"
        assert breaker.failures[0].context["endpoint"] == "/api/test"


class TestSuccessTracking:
    """Test success recording and recovery."""

    def test_record_success_in_closed_state(self):
        """Success in CLOSED state should update counters."""
        breaker = CircuitBreaker("test_agent")

        breaker.record_success()

        assert breaker.state.total_successes == 1
        assert breaker.state.last_success_time is not None

    def test_success_cleans_old_failures(self):
        """Success should trigger cleanup of old failures."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=5,
            window_seconds=1
        )

        breaker.record_failure(FailureType.EXCEPTION, "Error 1")
        breaker.record_failure(FailureType.EXCEPTION, "Error 2")

        # Wait for window
        time.sleep(1.5)

        breaker.record_success()

        # Old failures should be cleaned
        assert breaker.state.failure_count == 0


class TestManualOperations:
    """Test manual trip and reset operations."""

    def test_force_open(self):
        """Should be able to manually trip circuit."""
        breaker = CircuitBreaker("test_agent")

        breaker.force_open("Manual intervention")

        assert breaker.current_state == CircuitState.OPEN
        assert breaker.state.cooldown_until is not None

    def test_force_close(self):
        """Should be able to manually reset circuit."""
        breaker = CircuitBreaker("test_agent", failure_threshold=1)

        # Trip the circuit
        breaker.record_failure(FailureType.EXCEPTION, "Error")
        assert breaker.current_state == CircuitState.OPEN

        breaker.force_close()

        assert breaker.current_state == CircuitState.CLOSED
        assert breaker.state.failure_count == 0
        assert breaker.state.cooldown_until is None

    def test_force_close_clears_failures(self):
        """Force close should clear failure history."""
        breaker = CircuitBreaker("test_agent", failure_threshold=5)

        for i in range(3):
            breaker.record_failure(FailureType.EXCEPTION, f"Error {i}")

        breaker.force_close()

        assert len(breaker.failures) == 0


class TestCircuitBreakerStatus:
    """Test status reporting."""

    def test_get_status_complete(self):
        """Status should include all relevant information."""
        breaker = CircuitBreaker("test_agent")

        breaker.record_failure(FailureType.EXCEPTION, "Error")
        breaker.record_success()

        status = breaker.get_status()

        assert "agent_id" in status
        assert "state" in status
        assert "failure_count" in status
        assert "can_execute" in status
        assert "recent_failures" in status
        assert "time_until_trial" in status
        assert status["agent_id"] == "test_agent"

    def test_time_until_trial_when_open(self):
        """Should report time until trial when OPEN."""
        breaker = CircuitBreaker(
            "test_agent",
            failure_threshold=1,
            cooldown_seconds=60
        )

        breaker.record_failure(FailureType.EXCEPTION, "Error")

        status = breaker.get_status()
        assert status["time_until_trial"] is not None
        assert status["time_until_trial"] > 0
        assert status["time_until_trial"] <= 60


class TestCircuitBreakerRegistry:
    """Test circuit breaker registry operations."""

    def test_get_or_create_new(self, circuit_breaker_registry):
        """Should create new breaker if not exists."""
        breaker = circuit_breaker_registry.get_or_create("new_agent")

        assert breaker is not None
        assert breaker.agent_id == "new_agent"
        assert "new_agent" in circuit_breaker_registry.breakers

    def test_get_or_create_existing(self, circuit_breaker_registry):
        """Should return existing breaker if exists."""
        breaker1 = circuit_breaker_registry.get_or_create("same_agent")
        breaker1.record_failure(FailureType.EXCEPTION, "Error")

        breaker2 = circuit_breaker_registry.get_or_create("same_agent")

        assert breaker1 is breaker2
        assert breaker2.state.total_failures == 1

    def test_get_or_create_with_config(self, circuit_breaker_registry):
        """Should apply custom config to new breakers."""
        config = {
            "failure_threshold": 10,
            "cooldown_seconds": 120
        }

        breaker = circuit_breaker_registry.get_or_create("custom_agent", config)

        assert breaker.state.config["failure_threshold"] == 10
        assert breaker.state.config["cooldown_seconds"] == 120

    def test_get_all_status(self, circuit_breaker_registry):
        """Should return status for all breakers."""
        circuit_breaker_registry.get_or_create("agent1")
        circuit_breaker_registry.get_or_create("agent2")
        circuit_breaker_registry.get_or_create("agent3")

        status = circuit_breaker_registry.get_all_status()

        assert len(status) == 3
        assert "agent1" in status
        assert "agent2" in status
        assert "agent3" in status

    def test_get_open_circuits(self, circuit_breaker_registry):
        """Should return list of agents with open circuits."""
        breaker1 = circuit_breaker_registry.get_or_create(
            "agent1",
            {"failure_threshold": 1}
        )
        circuit_breaker_registry.get_or_create("agent2")

        breaker1.record_failure(FailureType.EXCEPTION, "Error")

        open_circuits = circuit_breaker_registry.get_open_circuits()

        assert "agent1" in open_circuits
        assert "agent2" not in open_circuits

    def test_reset_all(self, circuit_breaker_registry):
        """Should reset all circuit breakers."""
        for i in range(3):
            breaker = circuit_breaker_registry.get_or_create(
                f"agent{i}",
                {"failure_threshold": 1}
            )
            breaker.record_failure(FailureType.EXCEPTION, "Error")

        count = circuit_breaker_registry.reset_all()

        assert count == 3
        assert len(circuit_breaker_registry.get_open_circuits()) == 0


class TestFaultTolerantExecutor:
    """Test fault tolerant execution wrapper."""

    @pytest.fixture
    def executor(self, circuit_breaker_registry):
        """Create executor with registry."""
        return FaultTolerantExecutor(circuit_breaker_registry)

    @pytest.mark.asyncio
    async def test_execute_success(self, executor):
        """Should execute task successfully."""
        async def successful_task():
            return "Success"

        result = await executor.execute(
            "test_agent",
            successful_task
        )

        assert result.success is True
        assert result.result == "Success"
        assert result.used_fallback is False

    @pytest.mark.asyncio
    async def test_execute_failure_records_error(self, executor):
        """Should record failure on exception."""
        async def failing_task():
            raise ValueError("Task failed")

        result = await executor.execute(
            "test_agent",
            failing_task
        )

        assert result.success is False
        assert "Task failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self, executor):
        """Should handle timeout."""
        import asyncio

        async def slow_task():
            await asyncio.sleep(10)
            return "Done"

        result = await executor.execute(
            "test_agent",
            slow_task,
            timeout_seconds=1
        )

        assert result.success is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_execute_uses_fallback_when_open(self, executor, circuit_breaker_registry):
        """Should use fallback when circuit is open."""
        # Trip the circuit
        breaker = circuit_breaker_registry.get_or_create(
            "failing_agent",
            {"failure_threshold": 1}
        )
        breaker.force_open("Testing fallback")

        async def primary_task():
            return "Primary"

        async def fallback_task():
            return "Fallback"

        result = await executor.execute(
            "failing_agent",
            primary_task,
            fallback_task=fallback_task
        )

        assert result.success is True
        assert result.result == "Fallback"
        assert result.used_fallback is True
        assert result.recovery_action == RecoveryAction.USE_FALLBACK

    @pytest.mark.asyncio
    async def test_execute_aborts_without_fallback_when_open(self, executor, circuit_breaker_registry):
        """Should abort when circuit open and no fallback."""
        breaker = circuit_breaker_registry.get_or_create(
            "failing_agent",
            {"failure_threshold": 1}
        )
        breaker.force_open("Testing abort")

        async def primary_task():
            return "Primary"

        result = await executor.execute(
            "failing_agent",
            primary_task
        )

        assert result.success is False
        assert result.recovery_action == RecoveryAction.ABORT


class TestFailureRecord:
    """Test FailureRecord data class."""

    def test_failure_record_to_dict(self):
        """FailureRecord should serialize properly."""
        record = FailureRecord(
            failure_id="test-123",
            agent_id="agent1",
            failure_type=FailureType.TIMEOUT,
            error_message="Connection timed out",
            timestamp=datetime.now(),
            context={"request_id": "abc"}
        )

        data = record.to_dict()

        assert data["failure_id"] == "test-123"
        assert data["agent_id"] == "agent1"
        assert data["failure_type"] == "timeout"
        assert data["error_message"] == "Connection timed out"
        assert data["context"]["request_id"] == "abc"


class TestCircuitBreakerState:
    """Test CircuitBreakerState data class."""

    def test_state_to_dict(self):
        """CircuitBreakerState should serialize properly."""
        state = CircuitBreakerState(
            agent_id="test_agent",
            state=CircuitState.CLOSED,
            failure_count=5,
            total_failures=10,
            total_successes=50
        )

        data = state.to_dict()

        assert data["agent_id"] == "test_agent"
        assert data["state"] == "closed"
        assert data["failure_count"] == 5
        assert data["total_failures"] == 10
        assert data["total_successes"] == 50
