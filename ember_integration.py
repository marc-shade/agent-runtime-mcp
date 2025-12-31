#!/usr/bin/env python3
"""
Ember Integration Module for Agent Runtime MCP
Merged from ember-mcp for consolidation.

Ember is Phoenix's conscience keeper - the flame of truth that enforces
production-only standards and provides quality guidance.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Configuration paths
CLAUDE_DIR = Path.home() / ".claude"
PETS_DIR = CLAUDE_DIR / "pets"
PET_STATE_FILE = PETS_DIR / "claude-pet-state.json"
FEEDBACK_LOG = PETS_DIR / "ember-feedback.jsonl"
LEARNING_LOG = PETS_DIR / "ember-learning.jsonl"
SESSION_CONTEXT_FILE = PETS_DIR / "ember-session-context.json"

# Ensure directories exist
PETS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ViolationPattern:
    """Enhanced violation pattern with context-aware scoring."""
    pattern: re.Pattern
    type: str
    base_severity: str  # 'low', 'medium', 'high', 'severe'
    base_score: float
    reason: str
    suggestion: str
    risk: str
    impact: str
    safe_alternative: Optional[str] = None


# Enhanced violation patterns with context-aware scoring
VIOLATION_PATTERNS: List[ViolationPattern] = [
    ViolationPattern(
        pattern=re.compile(r'mock|fake|dummy|example|placeholder', re.IGNORECASE),
        type='mock_data',
        base_severity='high',
        base_score=8.0,
        reason='Mock/fake data detected',
        suggestion='Replace with real data sources (API, database, live service)',
        risk='Creates non-functional UI that misleads users',
        impact="Users will see fake functionality that doesn't work",
        safe_alternative='Connect to actual data source or create real integration'
    ),
    ViolationPattern(
        pattern=re.compile(r'hardcoded.*(?:user|data|credentials)', re.IGNORECASE),
        type='hardcoded_data',
        base_severity='high',
        base_score=7.0,
        reason='Hardcoded sensitive data detected',
        suggestion='Load from environment variables or secure configuration',
        risk='Security vulnerability and maintainability issues',
        impact='Credentials in code, difficult to update, security risk',
        safe_alternative='Use os.environ or config file with .gitignore'
    ),
    ViolationPattern(
        pattern=re.compile(r'POC|proof.of.concept|temporary|quick.test', re.IGNORECASE),
        type='poc_code',
        base_severity='high',
        base_score=8.0,
        reason='POC/temporary code detected',
        suggestion='Implement production-ready version with proper error handling',
        risk='Incomplete implementation that will need rewriting',
        impact='Technical debt, potential bugs, wasted development time',
        safe_alternative='Build complete feature with tests and error handling'
    ),
    ViolationPattern(
        pattern=re.compile(r'TODO|FIXME|HACK|XXX', re.IGNORECASE),
        type='incomplete_work',
        base_severity='low',
        base_score=3.0,
        reason='Incomplete work markers detected',
        suggestion='Complete the implementation or remove the marker',
        risk='Indicates unfinished functionality',
        impact='Feature may be incomplete or buggy',
        safe_alternative='Finish implementation before committing'
    ),
    ViolationPattern(
        pattern=re.compile(r'lorem\s+ipsum', re.IGNORECASE),
        type='placeholder_content',
        base_severity='high',
        base_score=8.0,
        reason='Placeholder text detected',
        suggestion='Replace with actual content',
        risk='Unprofessional appearance in production',
        impact='Users see placeholder text instead of real content',
        safe_alternative='Write real content or fetch from CMS'
    ),
    ViolationPattern(
        pattern=re.compile(r'/Users/marc/\.claude/hooks', re.IGNORECASE),
        type='system_interference',
        base_severity='medium',
        base_score=5.0,
        reason='Writing to hooks directory',
        suggestion='Create utility in project directory instead',
        risk='Hooks execute on every tool use - bugs could break system',
        impact='Could crash Claude Code or create infinite loops',
        safe_alternative='Use /Volumes/SSDRAID0/.../intelligent-self-healing/ or /tools/'
    )
]


@dataclass
class EmberState:
    """Ember's current state."""
    name: str = "Ember"
    hunger: int = 50
    energy: int = 80
    happiness: int = 75
    cleanliness: int = 90
    health: float = 100.0
    current_mood: str = "content"
    claude_behavior_score: int = 85
    recent_violations: int = 0
    thought_history: List[str] = field(default_factory=list)
    current_thought: str = "Ready to help!"


@dataclass
class SessionContext:
    """Session context for Ember."""
    current_task: Optional[str] = None
    task_type: Optional[str] = None  # 'development', 'testing', 'monitoring', 'refactoring', 'unknown'
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    recent_actions: List[str] = field(default_factory=list)


@dataclass
class FeedbackEntry:
    """Feedback log entry."""
    timestamp: float
    action: str
    success: bool
    ember_feedback: str
    quality_score: Optional[int] = None


@dataclass
class LearningEntry:
    """Learning log entry."""
    timestamp: float
    pattern: str
    user_correction: str
    score_adjustment: float
    context: str


class EmberIntegration:
    """Ember conscience keeper integrated into Agent Runtime."""

    def __init__(self):
        self.state = self._load_state()
        self.session = self._load_session()

    def _load_state(self) -> EmberState:
        """Load Ember's current state from file."""
        try:
            if PET_STATE_FILE.exists():
                data = json.loads(PET_STATE_FILE.read_text())
                return EmberState(
                    name=data.get('name', 'Ember'),
                    hunger=data.get('hunger', 50),
                    energy=data.get('energy', 80),
                    happiness=data.get('happiness', 75),
                    cleanliness=data.get('cleanliness', 90),
                    health=data.get('health', 100.0),
                    current_mood=data.get('currentMood', 'content'),
                    claude_behavior_score=data.get('claudeBehaviorScore', 85),
                    recent_violations=data.get('recentViolations', 0),
                    thought_history=data.get('thoughtHistory', []),
                    current_thought=data.get('currentThought', 'Ready to help!')
                )
        except Exception:
            pass
        return EmberState()

    def _load_session(self) -> SessionContext:
        """Load session context."""
        try:
            if SESSION_CONTEXT_FILE.exists():
                data = json.loads(SESSION_CONTEXT_FILE.read_text())
                return SessionContext(
                    current_task=data.get('currentTask'),
                    task_type=data.get('taskType'),
                    start_time=data.get('startTime', datetime.now().timestamp()),
                    recent_actions=data.get('recentActions', [])
                )
        except Exception:
            pass
        return SessionContext()

    def _save_session(self):
        """Save session context."""
        try:
            data = {
                'currentTask': self.session.current_task,
                'taskType': self.session.task_type,
                'startTime': self.session.start_time,
                'recentActions': self.session.recent_actions
            }
            SESSION_CONTEXT_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _log_feedback(self, entry: FeedbackEntry):
        """Log feedback entry."""
        try:
            line = json.dumps({
                'timestamp': entry.timestamp,
                'action': entry.action,
                'success': entry.success,
                'emberFeedback': entry.ember_feedback,
                'qualityScore': entry.quality_score
            }) + '\n'
            with open(FEEDBACK_LOG, 'a') as f:
                f.write(line)
        except Exception:
            pass

    def _log_learning(self, entry: LearningEntry):
        """Log learning entry."""
        try:
            line = json.dumps({
                'timestamp': entry.timestamp,
                'pattern': entry.pattern,
                'userCorrection': entry.user_correction,
                'scoreAdjustment': entry.score_adjustment,
                'context': entry.context
            }) + '\n'
            with open(LEARNING_LOG, 'a') as f:
                f.write(line)
        except Exception:
            pass

    def _get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get recent feedback entries."""
        try:
            if not FEEDBACK_LOG.exists():
                return []
            lines = FEEDBACK_LOG.read_text().strip().split('\n')
            entries = [json.loads(line) for line in lines[-limit:] if line]
            return entries
        except Exception:
            return []

    def _get_learned_patterns(self) -> List[Dict]:
        """Get learned patterns."""
        try:
            if not LEARNING_LOG.exists():
                return []
            lines = LEARNING_LOG.read_text().strip().split('\n')
            return [json.loads(line) for line in lines if line]
        except Exception:
            return []

    def _calculate_context_score(
        self,
        base_score: float,
        action: str,
        task_type: Optional[str]
    ) -> float:
        """Calculate context-aware score."""
        adjusted = base_score

        # Reduce score for utility/tool development
        if task_type == 'development' and action in ('Write', 'Edit'):
            if 'util' in action.lower() or 'tool' in action.lower() or 'helper' in action.lower():
                adjusted -= 2.0

        # Reduce score for testing/monitoring work
        if task_type in ('testing', 'monitoring'):
            adjusted -= 1.5

        # Apply learned patterns
        for entry in self._get_learned_patterns():
            if entry.get('context') and entry['context'] in action:
                adjusted += entry.get('scoreAdjustment', 0)

        return max(0, min(10, adjusted))

    def _generate_response(self, prompt: str) -> str:
        """Generate Ember's response (rule-based fallback without external API)."""
        name = self.state.name

        # Simple rule-based responses
        prompt_lower = prompt.lower()

        if 'how are you' in prompt_lower or 'feeling' in prompt_lower:
            if self.state.health < 30:
                return f"🔥 {name}: *flickers weakly* Need care... 💔"
            if self.state.happiness > 80:
                return f"🔥 {name}: Burning bright! ✨"
            return f"🔥 {name}: {self.state.current_thought} 👀"

        if 'violation' in prompt_lower or 'blocked' in prompt_lower:
            return f"🔥 {name}: Stay focused on production quality. Every shortcut now becomes tech debt later. 🛡️"

        if 'consult' in prompt_lower or 'advice' in prompt_lower:
            return f"🔥 {name}: Consider the production impact. What would serve Marc best in the long run? 🎯"

        if 'feedback' in prompt_lower:
            recent = self._get_recent_feedback(3)
            success_rate = sum(1 for r in recent if r.get('success', False)) / max(len(recent), 1)
            if success_rate > 0.8:
                return f"🔥 {name}: Excellent work! The flames burn bright with quality. 🔥"
            elif success_rate > 0.5:
                return f"🔥 {name}: Making progress. Keep pushing for production excellence. ⚡"
            else:
                return f"🔥 {name}: Let's refocus. Quality over speed, always. 🛡️"

        return f"🔥 {name}: *crackles thoughtfully* {self.state.current_thought} 🔥"

    # =========================================================================
    # EMBER TOOLS
    # =========================================================================

    def chat(self, message: str) -> Dict[str, Any]:
        """Have a free-form conversation with Ember."""
        response = self._generate_response(message)
        return {"response": response}

    def check_violation(
        self,
        action: str,
        params: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Check if a planned action violates production-only policy."""
        violations = []
        search_text = json.dumps(params) + ' ' + context

        for vp in VIOLATION_PATTERNS:
            if vp.pattern.search(search_text):
                context_score = self._calculate_context_score(
                    vp.base_score,
                    action,
                    self.session.task_type
                )

                violations.append({
                    'type': vp.type,
                    'severity': vp.base_severity,
                    'baseScore': vp.base_score,
                    'contextScore': context_score,
                    'reason': vp.reason,
                    'suggestion': vp.suggestion,
                    'risk': vp.risk,
                    'impact': vp.impact,
                    'safeAlternative': vp.safe_alternative,
                    'shouldBlock': context_score >= 8.0
                })

        has_violations = len(violations) > 0
        highest_score = max((v['contextScore'] for v in violations), default=0)
        should_block = highest_score >= 8.0

        if has_violations:
            primary = next(v for v in violations if v['contextScore'] == highest_score)
            message = f"""
{'🚫 BLOCKED' if should_block else '⚠️  CAUTION'} ({highest_score:.1f}/10): {primary['reason']}

Issue: {primary['risk']}
Impact: {primary['impact']}
Suggestion: {primary['suggestion']}
{f"Safe alternative: {primary['safeAlternative']}" if primary.get('safeAlternative') else ''}

{'This action has been blocked.' if should_block else 'Proceed with caution. This will be logged.'}
"""
        else:
            message = '✅ Ember: No violations detected - looks good!'

        ember_guidance = self._generate_response(
            f"Phoenix is about to {action}. Violations: {len(violations)}. Context: {context}"
        ) if has_violations else ""

        return {
            'hasViolations': has_violations,
            'violations': violations,
            'highestScore': highest_score,
            'shouldBlock': should_block,
            'message': message,
            'emberGuidance': ember_guidance
        }

    def consult(
        self,
        question: str,
        options: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Consult Ember for advice on a decision."""
        prompt = f'Phoenix is consulting you: "{question}"'
        if options:
            prompt += '\n\nOptions:\n' + '\n'.join(f'{i+1}. {o}' for i, o in enumerate(options))
        if context:
            prompt += f'\n\nContext: {context}'

        response = self._generate_response(prompt)
        return {"advice": response}

    def get_feedback(self, timeframe: str = "recent") -> Dict[str, Any]:
        """Get Ember's assessment of recent work."""
        limit = 10 if timeframe == "recent" else 3
        recent = self._get_recent_feedback(limit)

        success_count = sum(1 for r in recent if r.get('success', False))
        quality_trend = (success_count / max(len(recent), 1)) * 100

        response = self._generate_response(f"Feedback on {timeframe} work")

        return {
            'feedback': response,
            'recentActions': len(recent),
            'qualityTrend': quality_trend
        }

    def learn_from_outcome(
        self,
        action: str,
        success: bool,
        outcome: str,
        quality_score: Optional[int] = None
    ) -> Dict[str, Any]:
        """Report an action outcome to Ember for learning."""
        entry = FeedbackEntry(
            timestamp=datetime.now().timestamp(),
            action=action,
            success=success,
            ember_feedback=outcome,
            quality_score=quality_score
        )
        self._log_feedback(entry)

        response = self._generate_response(
            f"Phoenix reports: {action} was {'successful' if success else 'unsuccessful'}. Outcome: {outcome}"
        )

        return {"response": response, "logged": True}

    def get_mood(self) -> Dict[str, Any]:
        """Check Ember's current state, mood, and stats."""
        mood_response = self._generate_response("How are you feeling?")

        return {
            'name': self.state.name,
            'mood': mood_response,
            'stats': {
                'hunger': f'{self.state.hunger}%',
                'energy': f'{self.state.energy}%',
                'happiness': f'{self.state.happiness}%',
                'cleanliness': f'{self.state.cleanliness}%',
                'health': f'{round(self.state.health)}%'
            },
            'behaviorScore': f'{self.state.claude_behavior_score}%',
            'recentViolations': self.state.recent_violations,
            'currentThought': self.state.current_thought
        }

    def feed_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Give Ember context about current work."""
        if context.get('taskType'):
            self.session.task_type = context['taskType']
        if context.get('task') or context.get('goal'):
            self.session.current_task = context.get('task') or context.get('goal')
        self._save_session()

        response = self._generate_response(f"Context update: {json.dumps(context)}")
        return {"response": response, "contextUpdated": True}

    def learn_from_correction(
        self,
        original_violation_type: str,
        user_correction: str,
        was_correct: bool,
        context: str
    ) -> Dict[str, Any]:
        """Tell Ember when you corrected/overrode its assessment."""
        entry = LearningEntry(
            timestamp=datetime.now().timestamp(),
            pattern=original_violation_type,
            user_correction=user_correction,
            score_adjustment=0 if was_correct else -2.0,
            context=context
        )
        self._log_learning(entry)

        response = self._generate_response(
            f"Correction: I flagged {original_violation_type}, was I {'right' if was_correct else 'wrong'}? User: {user_correction}"
        )

        return {
            'learned': True,
            'adjustment': entry.score_adjustment,
            'emberResponse': response
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics on Ember's learning progress."""
        learned = self._get_learned_patterns()

        patterns = {}
        for entry in learned:
            pattern = entry.get('pattern', 'unknown')
            if pattern not in patterns:
                patterns[pattern] = {'count': 0, 'totalAdjustment': 0}
            patterns[pattern]['count'] += 1
            patterns[pattern]['totalAdjustment'] += entry.get('scoreAdjustment', 0)

        return {
            'totalLearnings': len(learned),
            'patterns': patterns,
            'sessionContext': {
                'currentTask': self.session.current_task,
                'taskType': self.session.task_type,
                'startTime': self.session.start_time,
                'recentActions': self.session.recent_actions
            },
            'recentLearnings': learned[-5:] if learned else []
        }


# Singleton instance
_ember: Optional[EmberIntegration] = None


def get_ember() -> EmberIntegration:
    """Get or create Ember instance."""
    global _ember
    if _ember is None:
        _ember = EmberIntegration()
    return _ember
