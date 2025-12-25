#!/usr/bin/env python3
"""
Relay Race Orchestrator for Multi-Agent Execution

Orchestrates the execution of relay race pipelines:
- Spawns agents in sequence with proper context
- Enforces quality gates (99.9% sequential execution rule)
- Manages token budgets
- Handles failures with single-agent retry
- Tracks provenance through L-Score chain

Reference: God Agent White Paper Section 9.2 - 48-Agent Relay Race Flow
Key Rule: Orchestrator does NOT spawn Agent N+1 until Agent N confirms storage
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

from relay_protocol import (
    RelayBaton, RelayPipeline, RelayResult, AgentSpec, RelayStatus,
    RelayProtocolManager, generate_agent_prompt,
    DEFAULT_QUALITY_THRESHOLD, DEFAULT_L_SCORE_THRESHOLD, MAX_RETRIES
)

logger = logging.getLogger("relay-orchestrator")


# ============================================================================
# ORCHESTRATOR TYPES
# ============================================================================

class ExecutionStrategy(Enum):
    """How to execute the relay pipeline."""
    SEQUENTIAL = "sequential"   # One agent at a time (default, 99.9% rule)
    PARALLEL_SAFE = "parallel_safe"  # Parallel where dependencies allow
    BATCH = "batch"             # Batch similar agents together


@dataclass
class ExecutionConfig:
    """Configuration for pipeline execution."""
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_concurrent_agents: int = 1  # For sequential, always 1
    timeout_per_step_seconds: int = 300  # 5 minutes default
    quality_gate_enabled: bool = True
    l_score_gate_enabled: bool = True
    auto_retry: bool = True
    max_retries: int = MAX_RETRIES
    save_intermediate_results: bool = True
    verbose_logging: bool = True


@dataclass
class StepExecution:
    """Tracks execution of a single step."""
    step_index: int
    agent_spec: AgentSpec
    baton: RelayBaton
    status: RelayStatus = RelayStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[RelayResult] = None
    error: Optional[str] = None
    retries: int = 0


@dataclass
class OrchestratorState:
    """Current state of orchestration."""
    pipeline_id: str
    current_step: int = 0
    total_steps: int = 0
    status: RelayStatus = RelayStatus.PENDING
    step_executions: List[StepExecution] = field(default_factory=list)
    tokens_used: int = 0
    quality_scores: List[float] = field(default_factory=list)
    l_scores: List[float] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ============================================================================
# AGENT EXECUTOR INTERFACE
# ============================================================================

class AgentExecutor:
    """
    Interface for executing agents. Subclass for different agent backends.

    Default implementation uses a callback pattern for flexibility.
    """

    def __init__(self):
        self.execution_callback: Optional[Callable[[str, RelayBaton], Awaitable[Dict[str, Any]]]] = None

    def set_callback(self, callback: Callable[[str, RelayBaton], Awaitable[Dict[str, Any]]]):
        """Set the callback function for agent execution."""
        self.execution_callback = callback

    async def execute(
        self,
        agent_spec: AgentSpec,
        baton: RelayBaton,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Execute an agent with the given baton.

        Args:
            agent_spec: Specification of the agent to execute
            baton: Relay baton with context and task

        Returns:
            Dict with:
            - success: bool
            - output_entity_id: int (if success)
            - quality_score: float
            - l_score: float
            - tokens_used: int
            - output_summary: str
            - error: str (if failure)
        """
        if self.execution_callback:
            try:
                result = await asyncio.wait_for(
                    self.execution_callback(agent_spec.agent_id, baton),
                    timeout=timeout_seconds
                )
                return result
            except asyncio.TimeoutError:
                return {
                    "success": False,
                    "error": f"Agent execution timed out after {timeout_seconds}s",
                    "quality_score": 0.0,
                    "l_score": 0.0,
                    "tokens_used": 0
                }
        else:
            # Default: Generate prompt and return for manual execution
            prompt = generate_agent_prompt(baton, agent_spec)
            return {
                "success": True,
                "prompt": prompt,
                "requires_manual_execution": True,
                "agent_id": agent_spec.agent_id,
                "output_key": baton.output_key
            }


class ClaudeAgentExecutor(AgentExecutor):
    """
    Executor that uses Claude Code Task tool for agent execution.

    Integrates with the existing Claude Code subagent system.
    """

    def __init__(self, anthropic_client=None):
        super().__init__()
        self.client = anthropic_client

    async def execute(
        self,
        agent_spec: AgentSpec,
        baton: RelayBaton,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """Execute via Claude Code Task tool pattern."""
        prompt = generate_agent_prompt(baton, agent_spec)

        # This returns the execution plan for Task tool
        return {
            "success": True,
            "execution_plan": {
                "subagent_type": self._map_agent_type(agent_spec.agent_type),
                "prompt": prompt,
                "description": f"Relay step {baton.sequence_position}: {agent_spec.agent_type}",
                "model": self._select_model(agent_spec)
            },
            "agent_id": agent_spec.agent_id,
            "output_key": baton.output_key,
            "baton": baton.to_dict()
        }

    def _map_agent_type(self, agent_type: str) -> str:
        """Map relay agent types to Claude Code subagent types."""
        mapping = {
            "researcher": "Explore",
            "analyzer": "Evidence-Based Analyzer",
            "synthesizer": "Integration Synthesizer",
            "validator": "Security Specialist",
            "formatter": "Documentation Scribe",
            "domain_expert": "LLXprt Domain Expert",
            "cross_validator": "code-reviewer"
        }
        return mapping.get(agent_type, "general-purpose")

    def _select_model(self, agent_spec: AgentSpec) -> str:
        """Select appropriate model based on agent capabilities."""
        if "validation" in agent_spec.capabilities or agent_spec.quality_threshold >= 0.8:
            return "sonnet"  # Higher quality for validation
        elif "research" in agent_spec.capabilities:
            return "haiku"   # Faster for research
        return "sonnet"      # Default


# ============================================================================
# RELAY ORCHESTRATOR
# ============================================================================

class RelayOrchestrator:
    """
    Orchestrates multi-agent relay race pipelines.

    Key responsibilities:
    1. Sequential execution with quality gates
    2. Baton passing with provenance tracking
    3. Failure handling and retry logic
    4. Progress monitoring and reporting
    """

    def __init__(
        self,
        protocol_manager: RelayProtocolManager,
        executor: Optional[AgentExecutor] = None,
        config: Optional[ExecutionConfig] = None
    ):
        self.protocol = protocol_manager
        self.executor = executor or AgentExecutor()
        self.config = config or ExecutionConfig()
        self.state: Optional[OrchestratorState] = None
        self._progress_callback: Optional[Callable[[OrchestratorState], None]] = None

    def set_progress_callback(self, callback: Callable[[OrchestratorState], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    async def execute_pipeline(
        self,
        pipeline_id: str,
        initial_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete relay race pipeline.

        Args:
            pipeline_id: ID of pipeline to execute
            initial_input: Optional initial context/input

        Returns:
            Complete execution results with all step outcomes
        """
        # Load pipeline
        pipeline_data = self.protocol.get_pipeline(pipeline_id)
        if not pipeline_data:
            return {"success": False, "error": f"Pipeline {pipeline_id} not found"}

        agents = [AgentSpec(**a) for a in pipeline_data["agents"]]

        # Initialize state
        self.state = OrchestratorState(
            pipeline_id=pipeline_id,
            total_steps=len(agents),
            started_at=datetime.now().isoformat()
        )

        logger.info(f"Starting relay pipeline {pipeline_id} with {len(agents)} agents")
        self._notify_progress()

        # Create initial baton
        initial_entity_id = initial_input.get("entity_id") if initial_input else None
        initial_key = initial_input.get("memory_key", "") if initial_input else ""
        initial_summary = initial_input.get("summary", "") if initial_input else ""

        try:
            current_baton = self.protocol.create_baton(
                pipeline_id=pipeline_id,
                step_index=0,
                task_description=agents[0].task_template,
                input_key=initial_key,
                input_summary=initial_summary,
                input_entity_id=initial_entity_id
            )
        except Exception as e:
            return {"success": False, "error": f"Failed to create initial baton: {e}"}

        # Execute each step sequentially
        for step_idx, agent_spec in enumerate(agents):
            self.state.current_step = step_idx
            self.state.status = RelayStatus.IN_PROGRESS

            step_exec = StepExecution(
                step_index=step_idx,
                agent_spec=agent_spec,
                baton=current_baton,
                started_at=datetime.now().isoformat()
            )
            self.state.step_executions.append(step_exec)
            self._notify_progress()

            # Execute step with retry logic
            result = await self._execute_step_with_retry(step_exec)

            if not result["success"]:
                # Step failed after retries
                self.state.status = RelayStatus.FAILED
                step_exec.status = RelayStatus.FAILED
                step_exec.error = result.get("error", "Unknown error")
                step_exec.completed_at = datetime.now().isoformat()

                logger.error(f"Pipeline {pipeline_id} failed at step {step_idx}: {step_exec.error}")
                self._notify_progress()

                return self._build_failure_response(step_idx, step_exec.error)

            # Step succeeded - update state
            step_exec.status = RelayStatus.PASSED
            step_exec.completed_at = datetime.now().isoformat()
            step_exec.result = RelayResult(
                step_index=step_idx,
                agent_id=agent_spec.agent_id,
                status=RelayStatus.PASSED,
                output_key=current_baton.output_key,
                output_entity_id=result.get("output_entity_id"),
                quality_score=result.get("quality_score", 0.0),
                l_score=result.get("l_score", 0.5),
                tokens_used=result.get("tokens_used", 0)
            )

            self.state.quality_scores.append(result.get("quality_score", 0.0))
            self.state.l_scores.append(result.get("l_score", 0.5))
            self.state.tokens_used += result.get("tokens_used", 0)

            # Pass baton to next agent (if not last step)
            if step_idx < len(agents) - 1:
                next_baton = self.protocol.pass_baton(
                    baton_id=current_baton.baton_id,
                    quality_score=result.get("quality_score", 0.0),
                    l_score=result.get("l_score", 0.5),
                    output_entity_id=result.get("output_entity_id", 0),
                    tokens_used=result.get("tokens_used", 0),
                    output_summary=result.get("output_summary", "")
                )

                if next_baton is None:
                    # Quality gate failed
                    self.state.status = RelayStatus.FAILED
                    return self._build_quality_gate_failure(step_idx, result)

                current_baton = next_baton
                logger.info(f"Baton passed to step {step_idx + 2}/{len(agents)}")

            self._notify_progress()

        # Pipeline completed successfully
        self.state.status = RelayStatus.COMPLETED
        self.state.completed_at = datetime.now().isoformat()

        # Final baton pass for completion
        self.protocol.pass_baton(
            baton_id=current_baton.baton_id,
            quality_score=result.get("quality_score", 0.0),
            l_score=result.get("l_score", 0.5),
            output_entity_id=result.get("output_entity_id", 0),
            tokens_used=result.get("tokens_used", 0),
            output_summary=result.get("output_summary", "")
        )

        logger.info(f"Pipeline {pipeline_id} completed successfully")
        self._notify_progress()

        return self._build_success_response()

    async def _execute_step_with_retry(
        self,
        step_exec: StepExecution
    ) -> Dict[str, Any]:
        """Execute a step with automatic retry on failure."""
        max_attempts = self.config.max_retries + 1 if self.config.auto_retry else 1

        for attempt in range(max_attempts):
            step_exec.retries = attempt

            if attempt > 0:
                logger.info(f"Retry {attempt}/{self.config.max_retries} for step {step_exec.step_index}")

            result = await self._execute_single_step(step_exec)

            if result["success"]:
                # Check quality gates
                if self.config.quality_gate_enabled:
                    if result.get("quality_score", 0) < step_exec.baton.quality_threshold:
                        logger.warning(f"Quality gate failed: {result.get('quality_score')} < {step_exec.baton.quality_threshold}")
                        if attempt < max_attempts - 1:
                            continue  # Retry
                        result["success"] = False
                        result["error"] = f"Quality gate failed after {max_attempts} attempts"

                if self.config.l_score_gate_enabled:
                    if result.get("l_score", 0) < step_exec.baton.l_score_threshold:
                        logger.warning(f"L-Score gate failed: {result.get('l_score')} < {step_exec.baton.l_score_threshold}")
                        if attempt < max_attempts - 1:
                            continue  # Retry
                        result["success"] = False
                        result["error"] = f"L-Score gate failed after {max_attempts} attempts"

                return result

            # Execution failed
            if attempt < max_attempts - 1:
                await asyncio.sleep(1)  # Brief delay before retry
                continue

        return result

    async def _execute_single_step(
        self,
        step_exec: StepExecution
    ) -> Dict[str, Any]:
        """Execute a single step."""
        start_time = time.time()

        try:
            result = await self.executor.execute(
                agent_spec=step_exec.agent_spec,
                baton=step_exec.baton,
                timeout_seconds=self.config.timeout_per_step_seconds
            )

            execution_time_ms = int((time.time() - start_time) * 1000)
            result["execution_time_ms"] = execution_time_ms

            if self.config.verbose_logging:
                logger.info(
                    f"Step {step_exec.step_index} ({step_exec.agent_spec.agent_type}) "
                    f"completed in {execution_time_ms}ms"
                )

            return result

        except Exception as e:
            logger.error(f"Step {step_exec.step_index} execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }

    def _notify_progress(self):
        """Notify progress callback if set."""
        if self._progress_callback and self.state:
            try:
                self._progress_callback(self.state)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _build_success_response(self) -> Dict[str, Any]:
        """Build success response."""
        return {
            "success": True,
            "pipeline_id": self.state.pipeline_id,
            "status": "completed",
            "total_steps": self.state.total_steps,
            "tokens_used": self.state.tokens_used,
            "average_quality": sum(self.state.quality_scores) / len(self.state.quality_scores) if self.state.quality_scores else 0,
            "average_l_score": sum(self.state.l_scores) / len(self.state.l_scores) if self.state.l_scores else 0.5,
            "step_results": [
                {
                    "step": se.step_index,
                    "agent": se.agent_spec.agent_type,
                    "status": se.status.value,
                    "quality": se.result.quality_score if se.result else 0,
                    "l_score": se.result.l_score if se.result else 0.5,
                    "output_key": se.result.output_key if se.result else None
                }
                for se in self.state.step_executions
            ],
            "started_at": self.state.started_at,
            "completed_at": self.state.completed_at
        }

    def _build_failure_response(self, failed_step: int, error: str) -> Dict[str, Any]:
        """Build failure response."""
        return {
            "success": False,
            "pipeline_id": self.state.pipeline_id,
            "status": "failed",
            "failed_at_step": failed_step,
            "total_steps": self.state.total_steps,
            "error": error,
            "completed_steps": failed_step,
            "tokens_used": self.state.tokens_used,
            "step_results": [
                {
                    "step": se.step_index,
                    "agent": se.agent_spec.agent_type,
                    "status": se.status.value,
                    "error": se.error
                }
                for se in self.state.step_executions
            ],
            "can_retry": failed_step > 0  # Can retry if not first step
        }

    def _build_quality_gate_failure(self, step: int, result: Dict) -> Dict[str, Any]:
        """Build quality gate failure response."""
        return {
            "success": False,
            "pipeline_id": self.state.pipeline_id,
            "status": "quality_gate_failed",
            "failed_at_step": step,
            "quality_score": result.get("quality_score", 0),
            "l_score": result.get("l_score", 0),
            "threshold": self.state.step_executions[step].baton.quality_threshold,
            "l_score_threshold": self.state.step_executions[step].baton.l_score_threshold,
            "recommendation": "Increase quality threshold or improve agent capabilities"
        }

    async def retry_from_step(
        self,
        pipeline_id: str,
        from_step: int
    ) -> Dict[str, Any]:
        """
        Retry pipeline from a specific step.

        Useful when manual intervention fixed an issue.
        """
        new_baton = self.protocol.retry_step(pipeline_id, from_step)

        if not new_baton:
            return {
                "success": False,
                "error": f"Cannot retry step {from_step}: max retries exceeded or step not found"
            }

        # Reset state from this step
        pipeline_data = self.protocol.get_pipeline(pipeline_id)
        agents = [AgentSpec(**a) for a in pipeline_data["agents"]]

        # Create partial state
        self.state = OrchestratorState(
            pipeline_id=pipeline_id,
            current_step=from_step,
            total_steps=len(agents),
            started_at=datetime.now().isoformat()
        )

        # Execute remaining steps
        return await self._execute_remaining_steps(agents[from_step:], new_baton, from_step)

    async def _execute_remaining_steps(
        self,
        remaining_agents: List[AgentSpec],
        starting_baton: RelayBaton,
        start_index: int
    ) -> Dict[str, Any]:
        """Execute remaining steps from a given point."""
        current_baton = starting_baton

        for offset, agent_spec in enumerate(remaining_agents):
            step_idx = start_index + offset
            self.state.current_step = step_idx

            step_exec = StepExecution(
                step_index=step_idx,
                agent_spec=agent_spec,
                baton=current_baton,
                started_at=datetime.now().isoformat()
            )
            self.state.step_executions.append(step_exec)

            result = await self._execute_step_with_retry(step_exec)

            if not result["success"]:
                self.state.status = RelayStatus.FAILED
                return self._build_failure_response(step_idx, result.get("error", "Unknown"))

            step_exec.status = RelayStatus.PASSED
            step_exec.completed_at = datetime.now().isoformat()

            self.state.quality_scores.append(result.get("quality_score", 0.0))
            self.state.l_scores.append(result.get("l_score", 0.5))
            self.state.tokens_used += result.get("tokens_used", 0)

            if offset < len(remaining_agents) - 1:
                next_baton = self.protocol.pass_baton(
                    baton_id=current_baton.baton_id,
                    quality_score=result.get("quality_score", 0.0),
                    l_score=result.get("l_score", 0.5),
                    output_entity_id=result.get("output_entity_id", 0),
                    tokens_used=result.get("tokens_used", 0),
                    output_summary=result.get("output_summary", "")
                )

                if not next_baton:
                    return self._build_quality_gate_failure(step_idx, result)

                current_baton = next_baton

        self.state.status = RelayStatus.COMPLETED
        self.state.completed_at = datetime.now().isoformat()
        return self._build_success_response()

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        if not self.state:
            return {"status": "no_active_execution"}

        return {
            "pipeline_id": self.state.pipeline_id,
            "status": self.state.status.value,
            "current_step": self.state.current_step,
            "total_steps": self.state.total_steps,
            "progress_percent": (self.state.current_step / self.state.total_steps * 100) if self.state.total_steps else 0,
            "tokens_used": self.state.tokens_used,
            "quality_scores": self.state.quality_scores,
            "l_scores": self.state.l_scores,
            "started_at": self.state.started_at
        }


# ============================================================================
# MCP TOOL REGISTRATION
# ============================================================================

def register_relay_tools(app, db_path: Path):
    """Register relay orchestrator MCP tools."""

    protocol_manager = RelayProtocolManager(db_path)
    orchestrator = RelayOrchestrator(protocol_manager)

    @app.tool()
    async def create_relay_pipeline(
        name: str,
        goal: str,
        agent_types: List[str],
        description: str = "",
        token_budget: int = 100000
    ) -> Dict[str, Any]:
        """
        Create a new relay race pipeline.

        The pipeline defines a sequence of agents that will execute in order,
        passing a baton with context between each step.

        Args:
            name: Pipeline name
            goal: High-level goal description
            agent_types: List of agent types in execution order
                Options: researcher, analyzer, synthesizer, validator, formatter, domain_expert
            description: Optional detailed description
            token_budget: Total token budget for pipeline

        Returns:
            Pipeline details including ID
        """
        from relay_protocol import STANDARD_AGENTS, AgentSpec

        # Convert agent types to specs
        agents = []
        for agent_type in agent_types:
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

        pipeline = protocol_manager.create_pipeline(
            name=name,
            goal=goal,
            agents=agents,
            description=description,
            token_budget=token_budget
        )

        return {
            "success": True,
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "goal": pipeline.goal,
            "agent_count": len(agents),
            "agents": [a.agent_type for a in agents],
            "token_budget": token_budget
        }

    @app.tool()
    async def get_relay_status(pipeline_id: str) -> Dict[str, Any]:
        """
        Get current status of a relay pipeline.

        Returns progress, quality scores, and step details.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Pipeline status and metrics
        """
        pipeline = protocol_manager.get_pipeline(pipeline_id)
        if not pipeline:
            return {"success": False, "error": f"Pipeline {pipeline_id} not found"}

        return {
            "success": True,
            **pipeline
        }

    @app.tool()
    async def advance_relay(
        pipeline_id: str,
        quality_score: float,
        l_score: float,
        output_entity_id: int,
        tokens_used: int,
        output_summary: str = ""
    ) -> Dict[str, Any]:
        """
        Manually advance relay to next step after completing current step.

        Use this after executing an agent step to pass the baton.

        Args:
            pipeline_id: Pipeline ID
            quality_score: Quality score of completed step (0.0-1.0)
            l_score: L-Score of output (0.0-1.0)
            output_entity_id: Entity ID where output was stored
            tokens_used: Tokens consumed by step
            output_summary: Summary for next agent

        Returns:
            Next baton or completion status
        """
        current_baton = protocol_manager.get_current_baton(pipeline_id)
        if not current_baton:
            return {"success": False, "error": "No active baton found for pipeline"}

        next_baton = protocol_manager.pass_baton(
            baton_id=current_baton.baton_id,
            quality_score=quality_score,
            l_score=l_score,
            output_entity_id=output_entity_id,
            tokens_used=tokens_used,
            output_summary=output_summary
        )

        if next_baton is None:
            # Check if completed or failed
            pipeline = protocol_manager.get_pipeline(pipeline_id)
            if pipeline["status"] == "completed":
                return {
                    "success": True,
                    "status": "pipeline_completed",
                    "message": "All steps completed successfully"
                }
            else:
                return {
                    "success": False,
                    "status": "quality_gate_failed",
                    "quality_score": quality_score,
                    "l_score": l_score
                }

        return {
            "success": True,
            "next_baton": next_baton.to_dict(),
            "next_step": next_baton.sequence_position,
            "total_steps": next_baton.total_agents
        }

    @app.tool()
    async def retry_relay_step(
        pipeline_id: str,
        step_index: int
    ) -> Dict[str, Any]:
        """
        Retry a failed relay step.

        Creates new baton for retry without restarting entire pipeline.

        Args:
            pipeline_id: Pipeline ID
            step_index: Step to retry (0-indexed)

        Returns:
            New baton for retry or error if max retries exceeded
        """
        new_baton = protocol_manager.retry_step(pipeline_id, step_index)

        if not new_baton:
            return {
                "success": False,
                "error": f"Cannot retry step {step_index}: max retries exceeded"
            }

        return {
            "success": True,
            "baton": new_baton.to_dict(),
            "retry_count": new_baton.retry_count
        }

    @app.tool()
    async def list_relay_pipelines(
        status: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        List relay pipelines.

        Args:
            status: Filter by status (pending, in_progress, completed, failed)
            limit: Maximum results

        Returns:
            List of pipelines with summary info
        """
        pipelines = protocol_manager.list_pipelines(status=status, limit=limit)

        return {
            "success": True,
            "count": len(pipelines),
            "pipelines": pipelines
        }

    @app.tool()
    async def get_relay_baton(pipeline_id: str) -> Dict[str, Any]:
        """
        Get current baton for a pipeline.

        Returns the baton with all context needed for next agent.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Current baton or error if none active
        """
        baton = protocol_manager.get_current_baton(pipeline_id)

        if not baton:
            return {
                "success": False,
                "error": "No active baton found"
            }

        return {
            "success": True,
            "baton": baton.to_dict(),
            "prompt": generate_agent_prompt(
                baton,
                AgentSpec(
                    agent_id="current",
                    agent_type="agent",
                    task_template=baton.task_description,
                    expected_output_type=baton.expected_output_type
                )
            )
        }

    logger.info("Relay orchestrator tools registered: 6 tools for pipeline management")
    return orchestrator
