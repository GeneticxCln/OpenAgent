"""
Memory and Learning System for OpenAgent.

This module provides conversation continuity across sessions, user preference learning,
command pattern recognition, and personalized suggestions based on usage history.
"""

import asyncio
import json
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openagent.core.enhanced_context import EnhancedContext
from openagent.core.exceptions import AgentError
from openagent.core.smart_prompts import TaskType


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    id: str
    session_id: str
    user_input: str
    agent_response: str
    task_type: str
    context_used: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    user_feedback: Optional[str] = None
    response_quality: Optional[float] = None  # 0.0 - 1.0


@dataclass
class ConversationSession:
    """A conversation session with multiple turns."""

    session_id: str
    start_time: float
    last_activity: float
    project_context: Optional[str] = None
    working_directory: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class UserPreference:
    """Individual user preference with learning metadata."""

    preference_type: str  # model_choice, verbosity, code_style, etc.
    preference_value: Any
    confidence: float = 1.0
    last_updated: float = field(default_factory=time.time)
    usage_count: int = 1
    context: Optional[str] = None  # Context where preference was learned


@dataclass
class CommandPattern:
    """Recognized command usage pattern."""

    pattern_id: str
    command_sequence: List[str]
    frequency: int
    success_rate: float
    avg_time_between_commands: float
    context_tags: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    user_efficiency_score: float = 0.0  # How efficient this pattern is


@dataclass
class PersonalizedSuggestion:
    """Personalized suggestion based on user patterns."""

    suggestion_id: str
    suggestion_text: str
    suggestion_type: str  # optimization, alternative, shortcut, etc.
    confidence: float
    relevance_score: float
    context_triggers: List[str] = field(default_factory=list)
    based_on_patterns: List[str] = field(default_factory=list)
    times_shown: int = 0
    times_accepted: int = 0


class MemoryManager:
    """
    Manages conversation history, user preferences, and learning data
    with persistent storage using SQLite.
    """

    def __init__(self, db_path: Optional[Path] = None, max_sessions: int = 100):
        self.db_path = db_path or Path.home() / ".openagent" / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions

        # In-memory caches for performance
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.user_preferences: Dict[str, UserPreference] = {}
        self.command_patterns: Dict[str, CommandPattern] = {}

        self._initialize_database()
        self._load_user_preferences()
        self._load_command_patterns()

    def _initialize_database(self):
        """Initialize SQLite database with required tables."""

        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    start_time REAL,
                    last_activity REAL,
                    project_context TEXT,
                    working_directory TEXT,
                    user_preferences TEXT,
                    is_active BOOLEAN
                );
                
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_input TEXT,
                    agent_response TEXT,
                    task_type TEXT,
                    context_used TEXT,
                    tools_used TEXT,
                    timestamp REAL,
                    user_feedback TEXT,
                    response_quality REAL,
                    FOREIGN KEY (session_id) REFERENCES conversations (session_id)
                );
                
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_type TEXT PRIMARY KEY,
                    preference_value TEXT,
                    confidence REAL,
                    last_updated REAL,
                    usage_count INTEGER,
                    context TEXT
                );
                
                CREATE TABLE IF NOT EXISTS command_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    command_sequence TEXT,
                    frequency INTEGER,
                    success_rate REAL,
                    avg_time_between_commands REAL,
                    context_tags TEXT,
                    last_seen REAL,
                    user_efficiency_score REAL
                );
                
                CREATE TABLE IF NOT EXISTS suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    suggestion_text TEXT,
                    suggestion_type TEXT,
                    confidence REAL,
                    relevance_score REAL,
                    context_triggers TEXT,
                    based_on_patterns TEXT,
                    times_shown INTEGER,
                    times_accepted INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_conversation_turns_session 
                    ON conversation_turns (session_id);
                CREATE INDEX IF NOT EXISTS idx_conversation_turns_timestamp 
                    ON conversation_turns (timestamp);
                CREATE INDEX IF NOT EXISTS idx_command_patterns_frequency 
                    ON command_patterns (frequency DESC);
            """
            )

    def _load_user_preferences(self):
        """Load user preferences from database into memory."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM user_preferences")
            for row in cursor.fetchall():
                pref = UserPreference(
                    preference_type=row[0],
                    preference_value=json.loads(row[1]),
                    confidence=row[2],
                    last_updated=row[3],
                    usage_count=row[4],
                    context=row[5],
                )
                self.user_preferences[pref.preference_type] = pref

    def _load_command_patterns(self):
        """Load command patterns from database into memory."""

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM command_patterns")
            for row in cursor.fetchall():
                pattern = CommandPattern(
                    pattern_id=row[0],
                    command_sequence=json.loads(row[1]),
                    frequency=row[2],
                    success_rate=row[3],
                    avg_time_between_commands=row[4],
                    context_tags=json.loads(row[5]),
                    last_seen=row[6],
                    user_efficiency_score=row[7],
                )
                self.command_patterns[pattern.pattern_id] = pattern

    async def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        context: Optional[EnhancedContext] = None,
    ) -> ConversationSession:
        """Get existing session or create a new one."""

        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_activity = time.time()
            return session

        # Generate new session ID if not provided
        if not session_id:
            session_id = f"session_{int(time.time() * 1000)}"

        # Create new session
        session = ConversationSession(
            session_id=session_id,
            start_time=time.time(),
            last_activity=time.time(),
            project_context=context.project.project_name if context else None,
            working_directory=(
                str(context.project.project_root)
                if context and context.project.project_root
                else None
            ),
        )

        self.active_sessions[session_id] = session
        await self._persist_session(session)

        return session

    async def add_conversation_turn(
        self,
        session: ConversationSession,
        user_input: str,
        agent_response: str,
        task_type: TaskType,
        context_used: Dict[str, Any],
        tools_used: List[str],
    ) -> ConversationTurn:
        """Add a new conversation turn to a session."""

        turn_id = f"turn_{int(time.time() * 1000000)}"

        turn = ConversationTurn(
            id=turn_id,
            session_id=session.session_id,
            user_input=user_input,
            agent_response=agent_response,
            task_type=task_type.value,
            context_used=context_used,
            tools_used=tools_used,
        )

        session.turns.append(turn)
        session.last_activity = time.time()

        await self._persist_conversation_turn(turn)
        return turn

    async def learn_user_preference(
        self,
        preference_type: str,
        preference_value: Any,
        context: Optional[str] = None,
        confidence_boost: float = 0.1,
    ):
        """Learn or update a user preference based on observed behavior."""

        if preference_type in self.user_preferences:
            pref = self.user_preferences[preference_type]
            pref.usage_count += 1
            pref.confidence = min(1.0, pref.confidence + confidence_boost)
            pref.last_updated = time.time()

            # Update value if it's different (weighted average for numeric values)
            if isinstance(preference_value, (int, float)) and isinstance(
                pref.preference_value, (int, float)
            ):
                weight = pref.confidence
                pref.preference_value = (
                    pref.preference_value * weight + preference_value
                ) / (weight + 1)
            else:
                pref.preference_value = preference_value
        else:
            pref = UserPreference(
                preference_type=preference_type,
                preference_value=preference_value,
                confidence=0.5,
                context=context,
            )
            self.user_preferences[preference_type] = pref

        await self._persist_user_preference(pref)

    async def analyze_command_patterns(
        self, commands: List[str], context_tags: List[str]
    ) -> List[CommandPattern]:
        """Analyze command sequences to identify patterns."""

        patterns_found = []

        # Look for sequences of 2-5 commands
        for seq_length in range(2, min(6, len(commands) + 1)):
            for i in range(len(commands) - seq_length + 1):
                sequence = commands[i : i + seq_length]
                pattern_id = f"pattern_{'_'.join(sequence[:3])}"  # Truncate for ID

                if pattern_id in self.command_patterns:
                    pattern = self.command_patterns[pattern_id]
                    pattern.frequency += 1
                    pattern.last_seen = time.time()
                    pattern.context_tags = list(
                        set(pattern.context_tags + context_tags)
                    )
                else:
                    pattern = CommandPattern(
                        pattern_id=pattern_id,
                        command_sequence=sequence,
                        frequency=1,
                        success_rate=1.0,  # Will be updated based on actual outcomes
                        avg_time_between_commands=5.0,  # Default estimate
                        context_tags=context_tags,
                    )
                    self.command_patterns[pattern_id] = pattern

                patterns_found.append(pattern)
                await self._persist_command_pattern(pattern)

        return patterns_found

    def get_conversation_context(
        self, session: ConversationSession, max_turns: int = 5
    ) -> List[Dict[str, str]]:
        """Get recent conversation context for continuity."""

        context = []
        recent_turns = session.turns[-max_turns:] if session.turns else []

        for turn in recent_turns:
            context.append({"role": "user", "content": turn.user_input})
            context.append({"role": "assistant", "content": turn.agent_response})

        return context

    def get_user_preference(self, preference_type: str, default: Any = None) -> Any:
        """Get a user preference value."""

        if preference_type in self.user_preferences:
            pref = self.user_preferences[preference_type]
            # Only return high-confidence preferences
            if pref.confidence > 0.6:
                return pref.preference_value

        return default

    def get_relevant_patterns(
        self, current_context: List[str], min_frequency: int = 3
    ) -> List[CommandPattern]:
        """Get command patterns relevant to current context."""

        relevant_patterns = []

        for pattern in self.command_patterns.values():
            if pattern.frequency >= min_frequency:
                # Check if pattern context tags overlap with current context
                overlap = set(pattern.context_tags) & set(current_context)
                if overlap:
                    relevant_patterns.append(pattern)

        # Sort by frequency and recency
        relevant_patterns.sort(key=lambda p: (p.frequency, p.last_seen), reverse=True)

        return relevant_patterns[:10]  # Top 10 most relevant

    async def generate_personalized_suggestions(
        self,
        context: EnhancedContext,
        current_task_type: TaskType,
        recent_commands: List[str],
    ) -> List[PersonalizedSuggestion]:
        """Generate personalized suggestions based on user patterns and preferences."""

        suggestions = []

        # Model preference suggestions
        preferred_model = self.get_user_preference("preferred_model")
        if preferred_model and context.project.project_type:
            suggestion = PersonalizedSuggestion(
                suggestion_id=f"model_pref_{int(time.time())}",
                suggestion_text=f"Based on your preferences, consider using {preferred_model} for this {context.project.project_type.value} project",
                suggestion_type="model_optimization",
                confidence=0.8,
                relevance_score=0.7,
                context_triggers=[context.project.project_type.value],
            )
            suggestions.append(suggestion)

        # Command pattern optimization suggestions
        context_tags = [
            context.project.project_type.value if context.project.project_type else "",
            current_task_type.value,
        ]
        context_tags = [tag for tag in context_tags if tag]  # Remove empty tags

        relevant_patterns = self.get_relevant_patterns(context_tags, min_frequency=2)

        for pattern in relevant_patterns[:3]:  # Top 3 patterns
            if len(recent_commands) >= 2:
                recent_sequence = recent_commands[-2:]

                # Check if user is following a pattern inefficiently
                if (
                    recent_sequence == pattern.command_sequence[:2]
                    and len(pattern.command_sequence) > 2
                ):
                    next_commands = pattern.command_sequence[2:]
                    suggestion = PersonalizedSuggestion(
                        suggestion_id=f"pattern_opt_{pattern.pattern_id}_{int(time.time())}",
                        suggestion_text=f"You might want to follow up with: {' && '.join(next_commands)}",
                        suggestion_type="workflow_optimization",
                        confidence=0.6,
                        relevance_score=0.8,
                        context_triggers=context_tags,
                        based_on_patterns=[pattern.pattern_id],
                    )
                    suggestions.append(suggestion)

        # Verbosity preference suggestions
        preferred_verbosity = self.get_user_preference("response_verbosity", "balanced")
        if preferred_verbosity == "concise" and current_task_type in [
            TaskType.TERMINAL_HELP,
            TaskType.SYSTEM_ADMIN,
        ]:
            suggestion = PersonalizedSuggestion(
                suggestion_id=f"verbosity_pref_{int(time.time())}",
                suggestion_text="I'll keep responses concise as you prefer. Ask for details if needed.",
                suggestion_type="response_style",
                confidence=0.7,
                relevance_score=0.6,
                context_triggers=[current_task_type.value],
            )
            suggestions.append(suggestion)

        # Error pattern learning suggestions
        if context.errors.error_patterns:
            common_errors = sorted(
                context.errors.error_patterns.items(), key=lambda x: x[1], reverse=True
            )
            if common_errors:
                error_type, count = common_errors[0]
                if count > 2:  # Recurring error
                    suggestion = PersonalizedSuggestion(
                        suggestion_id=f"error_pattern_{error_type}_{int(time.time())}",
                        suggestion_text=f"I notice you've had {count} {error_type.replace('_', ' ')} errors recently. Would you like tips to avoid these?",
                        suggestion_type="error_prevention",
                        confidence=0.8,
                        relevance_score=0.9,
                        context_triggers=[error_type],
                    )
                    suggestions.append(suggestion)

        # Git workflow suggestions
        if (
            context.git.is_repo
            and context.git.modified_files
            and len(context.git.modified_files) > 5
        ):
            suggestion = PersonalizedSuggestion(
                suggestion_id=f"git_workflow_{int(time.time())}",
                suggestion_text=f"You have {len(context.git.modified_files)} modified files. Consider committing in smaller, focused chunks for better history.",
                suggestion_type="workflow_improvement",
                confidence=0.6,
                relevance_score=0.7,
                context_triggers=["git", "many_modified_files"],
            )
            suggestions.append(suggestion)

        # Sort suggestions by relevance and confidence
        suggestions.sort(key=lambda s: s.relevance_score * s.confidence, reverse=True)

        return suggestions[:5]  # Return top 5 suggestions

    async def _persist_session(self, session: ConversationSession):
        """Persist session to database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations 
                (session_id, start_time, last_activity, project_context, working_directory, user_preferences, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session.session_id,
                    session.start_time,
                    session.last_activity,
                    session.project_context,
                    session.working_directory,
                    json.dumps(session.user_preferences),
                    session.is_active,
                ),
            )

    async def _persist_conversation_turn(self, turn: ConversationTurn):
        """Persist conversation turn to database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversation_turns 
                (id, session_id, user_input, agent_response, task_type, context_used, tools_used, timestamp, user_feedback, response_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    turn.id,
                    turn.session_id,
                    turn.user_input,
                    turn.agent_response,
                    turn.task_type,
                    json.dumps(turn.context_used),
                    json.dumps(turn.tools_used),
                    turn.timestamp,
                    turn.user_feedback,
                    turn.response_quality,
                ),
            )

    async def _persist_user_preference(self, pref: UserPreference):
        """Persist user preference to database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_preferences 
                (preference_type, preference_value, confidence, last_updated, usage_count, context)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    pref.preference_type,
                    json.dumps(pref.preference_value),
                    pref.confidence,
                    pref.last_updated,
                    pref.usage_count,
                    pref.context,
                ),
            )

    async def _persist_command_pattern(self, pattern: CommandPattern):
        """Persist command pattern to database."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO command_patterns 
                (pattern_id, command_sequence, frequency, success_rate, avg_time_between_commands, context_tags, last_seen, user_efficiency_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pattern.pattern_id,
                    json.dumps(pattern.command_sequence),
                    pattern.frequency,
                    pattern.success_rate,
                    pattern.avg_time_between_commands,
                    json.dumps(pattern.context_tags),
                    pattern.last_seen,
                    pattern.user_efficiency_score,
                ),
            )

    async def cleanup_old_sessions(self, max_age_days: int = 30):
        """Clean up old sessions and conversation turns."""

        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        with sqlite3.connect(self.db_path) as conn:
            # Delete old conversation turns
            conn.execute(
                """
                DELETE FROM conversation_turns 
                WHERE timestamp < ?
            """,
                (cutoff_time,),
            )

            # Delete old inactive sessions
            conn.execute(
                """
                DELETE FROM conversations 
                WHERE last_activity < ? AND is_active = 0
            """,
                (cutoff_time,),
            )

        # Clean up in-memory caches
        expired_sessions = [
            sid
            for sid, session in self.active_sessions.items()
            if session.last_activity < cutoff_time and not session.is_active
        ]

        for sid in expired_sessions:
            del self.active_sessions[sid]


class PersonalizedSuggestionEngine:
    """
    Engine for generating and managing personalized suggestions
    based on user behavior and patterns.
    """

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.suggestion_cache: Dict[str, List[PersonalizedSuggestion]] = {}
        self.cache_ttl = 300  # 5 minutes

    async def get_suggestions(
        self,
        context: EnhancedContext,
        task_type: TaskType,
        recent_commands: List[str],
        force_refresh: bool = False,
    ) -> List[PersonalizedSuggestion]:
        """Get personalized suggestions for the current context."""

        cache_key = f"{context.project.project_name}_{task_type.value}_{hash(tuple(recent_commands))}"

        if not force_refresh and cache_key in self.suggestion_cache:
            cached_suggestions = self.suggestion_cache[cache_key]
            if (
                cached_suggestions
                and time.time() - cached_suggestions[0].times_shown < self.cache_ttl
            ):
                return cached_suggestions

        suggestions = await self.memory_manager.generate_personalized_suggestions(
            context, task_type, recent_commands
        )

        # Update suggestion metrics
        for suggestion in suggestions:
            suggestion.times_shown += 1

        self.suggestion_cache[cache_key] = suggestions
        return suggestions

    async def record_suggestion_interaction(
        self, suggestion_id: str, accepted: bool, feedback: Optional[str] = None
    ):
        """Record user interaction with a suggestion for learning."""

        # Find suggestion in cache
        suggestion = None
        for cached_suggestions in self.suggestion_cache.values():
            for s in cached_suggestions:
                if s.suggestion_id == suggestion_id:
                    suggestion = s
                    break
            if suggestion:
                break

        if suggestion:
            if accepted:
                suggestion.times_accepted += 1
                suggestion.confidence = min(1.0, suggestion.confidence + 0.1)
            else:
                suggestion.confidence = max(0.1, suggestion.confidence - 0.05)

            # Learn from this interaction
            if feedback:
                await self._learn_from_suggestion_feedback(
                    suggestion, feedback, accepted
                )

    async def _learn_from_suggestion_feedback(
        self, suggestion: PersonalizedSuggestion, feedback: str, accepted: bool
    ):
        """Learn from user feedback on suggestions."""

        # Extract preference signals from feedback
        feedback_lower = feedback.lower()

        if "too verbose" in feedback_lower or "too long" in feedback_lower:
            await self.memory_manager.learn_user_preference(
                "response_verbosity", "concise", context=suggestion.suggestion_type
            )
        elif "more detail" in feedback_lower or "explain more" in feedback_lower:
            await self.memory_manager.learn_user_preference(
                "response_verbosity", "detailed", context=suggestion.suggestion_type
            )

        if (
            "different model" in feedback_lower
            and suggestion.suggestion_type == "model_optimization"
        ):
            # User didn't like the model suggestion, reduce confidence in current preference
            current_pref = self.memory_manager.get_user_preference("preferred_model")
            if current_pref:
                await self.memory_manager.learn_user_preference(
                    "preferred_model", current_pref, confidence_boost=-0.2
                )


# Utility functions for easy usage
async def get_memory_manager(db_path: Optional[Path] = None) -> MemoryManager:
    """Get or create a memory manager instance."""
    return MemoryManager(db_path)


async def create_learning_session(
    context: EnhancedContext, memory_manager: Optional[MemoryManager] = None
) -> Tuple[MemoryManager, ConversationSession]:
    """Create a new learning session with context."""

    if memory_manager is None:
        memory_manager = await get_memory_manager()

    session = await memory_manager.get_or_create_session(context=context)
    return memory_manager, session
