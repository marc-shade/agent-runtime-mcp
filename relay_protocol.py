#!/usr/bin/env python3
"""
Relay Race Protocol for Multi-Agent Orchestration

Implements the God Agent 48-agent Relay Race Protocol with:
- Structured baton passing between sequential agents
- Quality gate enforcement at each step
- Token budget tracking and enforcement
- Provenance chain preservation
- Fault tolerance with single-agent retry

Reference: God Agent White Paper Section 9.2 - 48-Agent Relay Race Flow
"""

import json
import sqlite3
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger("agent-runtime-relay")


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class RelayStatus(Enum):
    """Status of a relay pipeline or step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"  # Waiting for quality gate
    PASSED = "passed"    # Passed quality gate, ready for next
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DerivationMethod(Enum):
    """Methods for deriving knowledge between relay steps."""
    INFERENCE = "inference"       # Logical deduction
    EXTRACTION = "extraction"     # Data extraction
    SYNTHESIS = "synthesis"       # Combining multiple sources
    CITATION = "citation"         # Direct reference
    TRANSFORMATION = "transformation"  # Format/structure change
    VALIDATION = "validation"     # Verification step


# Default quality thresholds
DEFAULT_QUALITY_THRESHOLD = 0.6  # Minimum quality to pass baton
DEFAULT_L_SCORE_THRESHOLD = 0.3  # Minimum provenance quality
MAX_TOKEN_BUDGET = 100000        # Default max tokens per pipeline
MAX_RETRIES = 3                  # Default max retries per step


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RelayBaton:
    """
    Structured context passed between sequential agents in relay race.

    The baton contains everything an agent needs:
    - Where to get input (memory key)
    - What to do (task description)
    - Where to store output (output key)
    - Quality requirements
    - Token budgets
    - Provenance for L-Score tracking
    """
    baton_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Position in relay
    sequence_position: int = 1        # Agent N of M
    total_agents: int = 1

    # Context from previous agent
    input_key: str = ""               # Memory key to retrieve
    input_summary: str = ""           # Compressed summary (<500 tokens)
    input_entity_id: Optional[int] = None  # Entity ID for provenance

    # Task specification
    task_description: str = ""
    output_key: str = ""              # Memory key to store results
    expected_output_type: str = "analysis"  # Type of expected output

    # Quality requirements
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    l_score_threshold: float = DEFAULT_L_SCORE_THRESHOLD

    # Token budgeting
    max_output_tokens: int = 4000
    remaining_budget: int = MAX_TOKEN_BUDGET
    tokens_used: int = 0

    # Provenance chain (God Agent L-Score integration)
    source_chain: List[int] = field(default_factory=list)  # Entity IDs
    l_score_chain: List[float] = field(default_factory=list)
    derivation_method: str = "inference"

    # Fault tolerance
    retry_count: int = 0
    max_retries: int = MAX_RETRIES
    fallback_agent: Optional[str] = None

    # Metadata
    pipeline_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    passed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baton_id": self.baton_id,
            "sequence_position": self.sequence_position,
            "total_agents": self.total_agents,
            "input_key": self.input_key,
            "input_summary": self.input_summary,
            "input_entity_id": self.input_entity_id,
            "task_description": self.task_description,
            "output_key": self.output_key,
            "expected_output_type": self.expected_output_type,
            "quality_threshold": self.quality_threshold,
            "l_score_threshold": self.l_score_threshold,
            "max_output_tokens": self.max_output_tokens,
            "remaining_budget": self.remaining_budget,
            "tokens_used": self.tokens_used,
            "source_chain": self.source_chain,
            "l_score_chain": self.l_score_chain,
            "derivation_method": self.derivation_method,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "fallback_agent": self.fallback_agent,
            "pipeline_id": self.pipeline_id,
            "created_at": self.created_at,
            "passed_at": self.passed_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelayBaton":
        return cls(
            baton_id=data.get("baton_id", str(uuid.uuid4())[:8]),
            sequence_position=data.get("sequence_position", 1),
            total_agents=data.get("total_agents", 1),
            input_key=data.get("input_key", ""),
            input_summary=data.get("input_summary", ""),
            input_entity_id=data.get("input_entity_id"),
            task_description=data.get("task_description", ""),
            output_key=data.get("output_key", ""),
            expected_output_type=data.get("expected_output_type", "analysis"),
            quality_threshold=data.get("quality_threshold", DEFAULT_QUALITY_THRESHOLD),
            l_score_threshold=data.get("l_score_threshold", DEFAULT_L_SCORE_THRESHOLD),
            max_output_tokens=data.get("max_output_tokens", 4000),
            remaining_budget=data.get("remaining_budget", MAX_TOKEN_BUDGET),
            tokens_used=data.get("tokens_used", 0),
            source_chain=data.get("source_chain", []),
            l_score_chain=data.get("l_score_chain", []),
            derivation_method=data.get("derivation_method", "inference"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", MAX_RETRIES),
            fallback_agent=data.get("fallback_agent"),
            pipeline_id=data.get("pipeline_id", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            passed_at=data.get("passed_at")
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "RelayBaton":
        return cls.from_dict(json.loads(json_str))


@dataclass
class AgentSpec:
    """Specification for an agent in the relay pipeline."""
    agent_id: str
    agent_type: str               # e.g., "researcher", "analyzer", "validator"
    task_template: str            # Task description template
    expected_output_type: str     # e.g., "analysis", "summary", "validation"
    max_tokens: int = 4000
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    fallback_agent: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "task_template": self.task_template,
            "expected_output_type": self.expected_output_type,
            "max_tokens": self.max_tokens,
            "quality_threshold": self.quality_threshold,
            "fallback_agent": self.fallback_agent,
            "capabilities": self.capabilities
        }


@dataclass
class RelayResult:
    """Result from a single relay step execution."""
    step_index: int
    agent_id: str
    status: RelayStatus
    output_key: str
    output_entity_id: Optional[int] = None
    quality_score: float = 0.0
    l_score: float = 0.5
    tokens_used: int = 0
    execution_time_ms: int = 0
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "output_key": self.output_key,
            "output_entity_id": self.output_entity_id,
            "quality_score": self.quality_score,
            "l_score": self.l_score,
            "tokens_used": self.tokens_used,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "created_at": self.created_at
        }


@dataclass
class RelayPipeline:
    """
    A complete relay race pipeline definition.

    Contains all agents, their sequence, and execution state.
    """
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""
    description: str = ""
    goal: str = ""

    # Agent sequence
    agents: List[AgentSpec] = field(default_factory=list)
    current_step: int = 0

    # Execution state
    status: RelayStatus = RelayStatus.PENDING
    results: List[RelayResult] = field(default_factory=list)

    # Budget tracking
    total_token_budget: int = MAX_TOKEN_BUDGET
    tokens_used: int = 0

    # Quality tracking
    quality_scores: List[float] = field(default_factory=list)
    l_scores: List[float] = field(default_factory=list)

    # Provenance
    initial_entity_id: Optional[int] = None
    final_entity_id: Optional[int] = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "agents": [a.to_dict() for a in self.agents],
            "current_step": self.current_step,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "total_token_budget": self.total_token_budget,
            "tokens_used": self.tokens_used,
            "quality_scores": self.quality_scores,
            "l_scores": self.l_scores,
            "initial_entity_id": self.initial_entity_id,
            "final_entity_id": self.final_entity_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }

    @property
    def is_complete(self) -> bool:
        return self.status in (RelayStatus.COMPLETED, RelayStatus.FAILED, RelayStatus.CANCELLED)

    @property
    def progress_percent(self) -> float:
        if not self.agents:
            return 0.0
        return (self.current_step / len(self.agents)) * 100

    @property
    def average_quality(self) -> float:
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)

    @property
    def average_l_score(self) -> float:
        if not self.l_scores:
            return 0.5
        return sum(self.l_scores) / len(self.l_scores)


# ============================================================================
# AGENT PROMPT TEMPLATE
# ============================================================================

RELAY_AGENT_TEMPLATE = """
CONTEXT: Agent #{position}/{total} | Phase: {phase_name}
Pipeline: {pipeline_id} | Baton: {baton_id}

# BRIDGE (Critical - Do This First)
PREVIOUS: "{input_key}"
ACTION: Retrieve PREVIOUS memory immediately before proceeding.

{input_summary_section}

# MISSION
TASK: {task_description}
OUTPUT: Store results to "{output_key}"
EXPECTED TYPE: {expected_output_type}

# CONSTRAINTS
- Max output: {max_tokens} tokens
- Quality threshold: {quality_threshold}
- L-Score threshold: {l_score_threshold}
- Remaining budget: {remaining_budget} tokens

# PROVENANCE REQUIREMENTS
- Derivation method: {derivation_method}
- Source entities: {source_chain}
- Link to parent node (no orphans)
- Record L-Score sources for chain

# HANDOFF PROTOCOL
1. Execute task completely
2. Store output to "{output_key}" with provenance
3. Report output key and entity ID for next agent
4. Provide quality self-assessment (0.0-1.0)
5. Report tokens used for budget tracking

# QUALITY FEEDBACK
After completion, provide:
- quality_score: Your assessment of output quality (0.0-1.0)
- confidence: Confidence in the result (0.0-1.0)
- suggestions: Any notes for next agent
"""


def generate_agent_prompt(baton: RelayBaton, agent_spec: AgentSpec) -> str:
    """Generate the prompt for an agent from baton and spec."""

    input_summary_section = ""
    if baton.input_summary:
        input_summary_section = f"""
# INPUT SUMMARY (from previous agent)
{baton.input_summary}
"""

    return RELAY_AGENT_TEMPLATE.format(
        position=baton.sequence_position,
        total=baton.total_agents,
        phase_name=agent_spec.agent_type.title(),
        pipeline_id=baton.pipeline_id,
        baton_id=baton.baton_id,
        input_key=baton.input_key,
        input_summary_section=input_summary_section,
        task_description=baton.task_description or agent_spec.task_template,
        output_key=baton.output_key,
        expected_output_type=agent_spec.expected_output_type,
        max_tokens=agent_spec.max_tokens,
        quality_threshold=agent_spec.quality_threshold,
        l_score_threshold=baton.l_score_threshold,
        remaining_budget=baton.remaining_budget,
        derivation_method=baton.derivation_method,
        source_chain=baton.source_chain if baton.source_chain else "[Initial - No sources]"
    )


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def init_relay_schema(db_path: Path) -> None:
    """Initialize relay protocol database tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Pipelines table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relay_pipelines (
            pipeline_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            goal TEXT,
            agents_json TEXT,
            current_step INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            total_token_budget INTEGER DEFAULT 100000,
            tokens_used INTEGER DEFAULT 0,
            quality_scores_json TEXT DEFAULT '[]',
            l_scores_json TEXT DEFAULT '[]',
            initial_entity_id INTEGER,
            final_entity_id INTEGER,
            created_at TEXT,
            started_at TEXT,
            completed_at TEXT
        )
    """)

    # Results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relay_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pipeline_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            output_key TEXT,
            output_entity_id INTEGER,
            quality_score REAL DEFAULT 0.0,
            l_score REAL DEFAULT 0.5,
            tokens_used INTEGER DEFAULT 0,
            execution_time_ms INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TEXT,
            FOREIGN KEY (pipeline_id) REFERENCES relay_pipelines(pipeline_id)
        )
    """)

    # Batons table (current state tracking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS relay_batons (
            baton_id TEXT PRIMARY KEY,
            pipeline_id TEXT NOT NULL,
            sequence_position INTEGER NOT NULL,
            baton_json TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            passed_at TEXT,
            FOREIGN KEY (pipeline_id) REFERENCES relay_pipelines(pipeline_id)
        )
    """)

    # Indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relay_pipelines_status
        ON relay_pipelines(status)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relay_results_pipeline
        ON relay_results(pipeline_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relay_batons_pipeline
        ON relay_batons(pipeline_id)
    """)

    conn.commit()
    conn.close()
    logger.info("Relay protocol schema initialized")


class RelayProtocolManager:
    """
    Manages relay race protocol state and operations.

    Key responsibilities:
    - Pipeline creation and management
    - Baton creation and passing
    - Quality gate enforcement
    - Provenance chain tracking
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        init_relay_schema(db_path)

    def create_pipeline(
        self,
        name: str,
        goal: str,
        agents: List[AgentSpec],
        description: str = "",
        token_budget: int = MAX_TOKEN_BUDGET,
        initial_entity_id: Optional[int] = None
    ) -> RelayPipeline:
        """
        Create a new relay race pipeline.

        Args:
            name: Pipeline name
            goal: High-level goal description
            agents: Ordered list of agent specifications
            description: Optional detailed description
            token_budget: Total token budget for pipeline
            initial_entity_id: Starting entity ID for provenance

        Returns:
            Created RelayPipeline
        """
        pipeline = RelayPipeline(
            name=name,
            description=description,
            goal=goal,
            agents=agents,
            total_token_budget=token_budget,
            initial_entity_id=initial_entity_id
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO relay_pipelines
            (pipeline_id, name, description, goal, agents_json,
             total_token_budget, initial_entity_id, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pipeline.pipeline_id,
            pipeline.name,
            pipeline.description,
            pipeline.goal,
            json.dumps([a.to_dict() for a in agents]),
            token_budget,
            initial_entity_id,
            pipeline.created_at,
            pipeline.status.value
        ))

        conn.commit()
        conn.close()

        logger.info(f"Created relay pipeline '{name}' with {len(agents)} agents: {pipeline.pipeline_id}")
        return pipeline

    def create_baton(
        self,
        pipeline_id: str,
        step_index: int,
        task_description: str,
        input_key: str = "",
        input_summary: str = "",
        input_entity_id: Optional[int] = None,
        derivation_method: str = "inference"
    ) -> RelayBaton:
        """
        Create a baton for a specific pipeline step.

        Args:
            pipeline_id: Pipeline this baton belongs to
            step_index: Which step (0-indexed)
            task_description: Task for this step
            input_key: Memory key to retrieve from previous step
            input_summary: Summary of input context
            input_entity_id: Entity ID for provenance
            derivation_method: How this step derives from previous

        Returns:
            Created RelayBaton
        """
        # Get pipeline info
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT agents_json, total_token_budget, tokens_used,
                   quality_scores_json, l_scores_json
            FROM relay_pipelines WHERE pipeline_id = ?
        """, (pipeline_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Pipeline {pipeline_id} not found")

        agents_json, total_budget, tokens_used, quality_json, l_scores_json = row
        agents = json.loads(agents_json)
        quality_scores = json.loads(quality_json) if quality_json else []
        l_scores = json.loads(l_scores_json) if l_scores_json else []

        if step_index >= len(agents):
            conn.close()
            raise ValueError(f"Step {step_index} exceeds pipeline agent count {len(agents)}")

        agent = agents[step_index]

        # Build source chain from previous steps
        source_chain = []
        if input_entity_id:
            source_chain.append(input_entity_id)

        # Create output key
        output_key = f"relay_{pipeline_id}_step{step_index}_{agent['agent_type']}"

        baton = RelayBaton(
            pipeline_id=pipeline_id,
            sequence_position=step_index + 1,
            total_agents=len(agents),
            input_key=input_key,
            input_summary=input_summary,
            input_entity_id=input_entity_id,
            task_description=task_description or agent.get("task_template", ""),
            output_key=output_key,
            expected_output_type=agent.get("expected_output_type", "analysis"),
            quality_threshold=agent.get("quality_threshold", DEFAULT_QUALITY_THRESHOLD),
            max_output_tokens=agent.get("max_tokens", 4000),
            remaining_budget=total_budget - tokens_used,
            source_chain=source_chain,
            l_score_chain=l_scores.copy(),
            derivation_method=derivation_method,
            fallback_agent=agent.get("fallback_agent")
        )

        # Store baton
        cursor.execute("""
            INSERT INTO relay_batons
            (baton_id, pipeline_id, sequence_position, baton_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            baton.baton_id,
            pipeline_id,
            baton.sequence_position,
            baton.to_json(),
            RelayStatus.PENDING.value,
            baton.created_at
        ))

        conn.commit()
        conn.close()

        logger.info(f"Created baton {baton.baton_id} for step {step_index} in pipeline {pipeline_id}")
        return baton

    def pass_baton(
        self,
        baton_id: str,
        quality_score: float,
        l_score: float,
        output_entity_id: int,
        tokens_used: int,
        output_summary: str = ""
    ) -> Optional[RelayBaton]:
        """
        Pass baton to next agent after quality gate check.

        Args:
            baton_id: Current baton ID
            quality_score: Quality score of completed step
            l_score: L-Score of output
            output_entity_id: Entity ID of stored output
            tokens_used: Tokens consumed
            output_summary: Summary for next agent

        Returns:
            Next baton if quality gate passes and more steps remain,
            None if pipeline complete or quality gate failed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current baton
        cursor.execute("""
            SELECT baton_json, pipeline_id FROM relay_batons WHERE baton_id = ?
        """, (baton_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Baton {baton_id} not found")

        baton = RelayBaton.from_json(row[0])
        pipeline_id = row[1]

        # Quality gate check
        passed = quality_score >= baton.quality_threshold and l_score >= baton.l_score_threshold

        # Store result
        cursor.execute("""
            INSERT INTO relay_results
            (pipeline_id, step_index, agent_id, status, output_key, output_entity_id,
             quality_score, l_score, tokens_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pipeline_id,
            baton.sequence_position - 1,
            f"agent_step_{baton.sequence_position}",
            RelayStatus.PASSED.value if passed else RelayStatus.FAILED.value,
            baton.output_key,
            output_entity_id,
            quality_score,
            l_score,
            tokens_used,
            datetime.now().isoformat()
        ))

        # Update baton status
        baton.passed_at = datetime.now().isoformat()
        cursor.execute("""
            UPDATE relay_batons SET status = ?, passed_at = ?, baton_json = ?
            WHERE baton_id = ?
        """, (
            RelayStatus.PASSED.value if passed else RelayStatus.FAILED.value,
            baton.passed_at,
            baton.to_json(),
            baton_id
        ))

        # Update pipeline
        cursor.execute("""
            SELECT total_token_budget, tokens_used, quality_scores_json,
                   l_scores_json, agents_json
            FROM relay_pipelines WHERE pipeline_id = ?
        """, (pipeline_id,))

        pipeline_row = cursor.fetchone()
        total_budget, pipeline_tokens, quality_json, l_scores_json, agents_json = pipeline_row

        quality_scores = json.loads(quality_json) if quality_json else []
        l_scores = json.loads(l_scores_json) if l_scores_json else []
        agents = json.loads(agents_json)

        quality_scores.append(quality_score)
        l_scores.append(l_score)
        new_tokens = pipeline_tokens + tokens_used

        if not passed:
            # Quality gate failed
            cursor.execute("""
                UPDATE relay_pipelines SET
                    status = ?, tokens_used = ?,
                    quality_scores_json = ?, l_scores_json = ?
                WHERE pipeline_id = ?
            """, (
                RelayStatus.FAILED.value,
                new_tokens,
                json.dumps(quality_scores),
                json.dumps(l_scores),
                pipeline_id
            ))
            conn.commit()
            conn.close()
            logger.warning(f"Quality gate failed for baton {baton_id}: q={quality_score}, l={l_score}")
            return None

        # Check if pipeline complete
        next_step = baton.sequence_position  # 1-indexed, so this is next 0-indexed step
        if next_step >= len(agents):
            # Pipeline complete
            cursor.execute("""
                UPDATE relay_pipelines SET
                    status = ?, current_step = ?, tokens_used = ?,
                    quality_scores_json = ?, l_scores_json = ?,
                    final_entity_id = ?, completed_at = ?
                WHERE pipeline_id = ?
            """, (
                RelayStatus.COMPLETED.value,
                next_step,
                new_tokens,
                json.dumps(quality_scores),
                json.dumps(l_scores),
                output_entity_id,
                datetime.now().isoformat(),
                pipeline_id
            ))
            conn.commit()
            conn.close()
            logger.info(f"Pipeline {pipeline_id} completed successfully")
            return None

        # Update pipeline progress
        cursor.execute("""
            UPDATE relay_pipelines SET
                current_step = ?, tokens_used = ?,
                quality_scores_json = ?, l_scores_json = ?
            WHERE pipeline_id = ?
        """, (
            next_step,
            new_tokens,
            json.dumps(quality_scores),
            json.dumps(l_scores),
            pipeline_id
        ))

        conn.commit()
        conn.close()

        # Create next baton
        next_baton = self.create_baton(
            pipeline_id=pipeline_id,
            step_index=next_step,
            task_description="",  # Will use agent template
            input_key=baton.output_key,
            input_summary=output_summary,
            input_entity_id=output_entity_id,
            derivation_method="inference"
        )

        # Add previous L-Scores to chain
        next_baton.l_score_chain = l_scores
        next_baton.source_chain = list(set(baton.source_chain + [output_entity_id]))

        logger.info(f"Passed baton to step {next_step + 1} in pipeline {pipeline_id}")
        return next_baton

    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline status and details."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM relay_pipelines WHERE pipeline_id = ?
        """, (pipeline_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        columns = [desc[0] for desc in cursor.description]
        pipeline = dict(zip(columns, row))

        # Parse JSON fields
        pipeline["agents"] = json.loads(pipeline["agents_json"]) if pipeline.get("agents_json") else []
        pipeline["quality_scores"] = json.loads(pipeline["quality_scores_json"]) if pipeline.get("quality_scores_json") else []
        pipeline["l_scores"] = json.loads(pipeline["l_scores_json"]) if pipeline.get("l_scores_json") else []

        # Get results
        cursor.execute("""
            SELECT * FROM relay_results WHERE pipeline_id = ? ORDER BY step_index
        """, (pipeline_id,))

        result_columns = [desc[0] for desc in cursor.description]
        pipeline["results"] = [dict(zip(result_columns, r)) for r in cursor.fetchall()]

        conn.close()

        # Calculate summary
        pipeline["progress_percent"] = (pipeline["current_step"] / len(pipeline["agents"]) * 100) if pipeline["agents"] else 0
        pipeline["average_quality"] = sum(pipeline["quality_scores"]) / len(pipeline["quality_scores"]) if pipeline["quality_scores"] else 0
        pipeline["average_l_score"] = sum(pipeline["l_scores"]) / len(pipeline["l_scores"]) if pipeline["l_scores"] else 0.5

        return pipeline

    def get_current_baton(self, pipeline_id: str) -> Optional[RelayBaton]:
        """Get the current active baton for a pipeline."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT baton_json FROM relay_batons
            WHERE pipeline_id = ? AND status = ?
            ORDER BY sequence_position DESC LIMIT 1
        """, (pipeline_id, RelayStatus.PENDING.value))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return RelayBaton.from_json(row[0])

    def retry_step(self, pipeline_id: str, step_index: int) -> Optional[RelayBaton]:
        """
        Retry a failed step (single-agent retry, not full pipeline restart).

        Returns new baton if retry allowed, None if max retries exceeded.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get previous baton for this step
        cursor.execute("""
            SELECT baton_json FROM relay_batons
            WHERE pipeline_id = ? AND sequence_position = ?
            ORDER BY created_at DESC LIMIT 1
        """, (pipeline_id, step_index + 1))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        old_baton = RelayBaton.from_json(row[0])

        if old_baton.retry_count >= old_baton.max_retries:
            logger.warning(f"Max retries ({old_baton.max_retries}) exceeded for step {step_index}")
            conn.close()
            return None

        conn.close()

        # Create new baton with incremented retry count
        new_baton = self.create_baton(
            pipeline_id=pipeline_id,
            step_index=step_index,
            task_description=old_baton.task_description,
            input_key=old_baton.input_key,
            input_summary=old_baton.input_summary,
            input_entity_id=old_baton.input_entity_id,
            derivation_method=old_baton.derivation_method
        )
        new_baton.retry_count = old_baton.retry_count + 1
        new_baton.source_chain = old_baton.source_chain
        new_baton.l_score_chain = old_baton.l_score_chain

        logger.info(f"Created retry baton (attempt {new_baton.retry_count}) for step {step_index}")
        return new_baton

    def list_pipelines(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List pipelines, optionally filtered by status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            cursor.execute("""
                SELECT pipeline_id, name, goal, status, current_step,
                       tokens_used, created_at, completed_at, agents_json
                FROM relay_pipelines
                WHERE status = ?
                ORDER BY created_at DESC LIMIT ?
            """, (status, limit))
        else:
            cursor.execute("""
                SELECT pipeline_id, name, goal, status, current_step,
                       tokens_used, created_at, completed_at, agents_json
                FROM relay_pipelines
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))

        results = []
        for row in cursor.fetchall():
            agents = json.loads(row[8]) if row[8] else []
            results.append({
                "pipeline_id": row[0],
                "name": row[1],
                "goal": row[2],
                "status": row[3],
                "current_step": row[4],
                "total_steps": len(agents),
                "tokens_used": row[5],
                "created_at": row[6],
                "completed_at": row[7],
                "progress_percent": (row[4] / len(agents) * 100) if agents else 0
            })

        conn.close()
        return results


# ============================================================================
# PREDEFINED AGENT TEMPLATES
# ============================================================================

# Standard agent types for common relay patterns
STANDARD_AGENTS = {
    "researcher": AgentSpec(
        agent_id="researcher",
        agent_type="researcher",
        task_template="Research and gather relevant information on the topic. Cite sources and assess credibility.",
        expected_output_type="research_findings",
        max_tokens=6000,
        quality_threshold=0.6,
        capabilities=["web_search", "memory_search", "document_analysis"]
    ),
    "analyzer": AgentSpec(
        agent_id="analyzer",
        agent_type="analyzer",
        task_template="Analyze the gathered information. Identify patterns, insights, and key findings.",
        expected_output_type="analysis",
        max_tokens=5000,
        quality_threshold=0.7,
        capabilities=["pattern_detection", "reasoning", "synthesis"]
    ),
    "synthesizer": AgentSpec(
        agent_id="synthesizer",
        agent_type="synthesizer",
        task_template="Synthesize findings into coherent conclusions. Highlight main takeaways.",
        expected_output_type="synthesis",
        max_tokens=4000,
        quality_threshold=0.7,
        capabilities=["summarization", "integration", "conclusion_drawing"]
    ),
    "validator": AgentSpec(
        agent_id="validator",
        agent_type="validator",
        task_template="Validate conclusions against sources. Check for logical consistency and factual accuracy.",
        expected_output_type="validation_report",
        max_tokens=3000,
        quality_threshold=0.8,
        capabilities=["fact_checking", "logic_validation", "source_verification"]
    ),
    "formatter": AgentSpec(
        agent_id="formatter",
        agent_type="formatter",
        task_template="Format the validated content for final delivery. Ensure clarity and completeness.",
        expected_output_type="formatted_output",
        max_tokens=4000,
        quality_threshold=0.6,
        capabilities=["formatting", "structuring", "presentation"]
    )
}


def create_standard_research_pipeline(
    manager: RelayProtocolManager,
    goal: str,
    depth: str = "standard"
) -> RelayPipeline:
    """
    Create a standard research pipeline.

    Depth options:
    - "quick": 3 agents (research, synthesize, format)
    - "standard": 5 agents (research, analyze, synthesize, validate, format)
    - "deep": 7+ agents (multiple research phases, cross-validation)
    """
    if depth == "quick":
        agents = [
            STANDARD_AGENTS["researcher"],
            STANDARD_AGENTS["synthesizer"],
            STANDARD_AGENTS["formatter"]
        ]
    elif depth == "deep":
        agents = [
            STANDARD_AGENTS["researcher"],
            AgentSpec(
                agent_id="domain_expert",
                agent_type="domain_expert",
                task_template="Apply domain expertise to research findings. Add specialized insights.",
                expected_output_type="expert_analysis",
                max_tokens=5000,
                quality_threshold=0.7
            ),
            STANDARD_AGENTS["analyzer"],
            AgentSpec(
                agent_id="cross_validator",
                agent_type="cross_validator",
                task_template="Cross-validate findings against multiple sources. Flag contradictions.",
                expected_output_type="validation_matrix",
                max_tokens=4000,
                quality_threshold=0.8
            ),
            STANDARD_AGENTS["synthesizer"],
            STANDARD_AGENTS["validator"],
            STANDARD_AGENTS["formatter"]
        ]
    else:  # standard
        agents = [
            STANDARD_AGENTS["researcher"],
            STANDARD_AGENTS["analyzer"],
            STANDARD_AGENTS["synthesizer"],
            STANDARD_AGENTS["validator"],
            STANDARD_AGENTS["formatter"]
        ]

    return manager.create_pipeline(
        name=f"Research Pipeline ({depth})",
        goal=goal,
        agents=agents,
        description=f"Standard {depth} research pipeline: {' -> '.join([a.agent_type for a in agents])}"
    )
