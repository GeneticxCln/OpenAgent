"""
History Intelligence for OpenAgent.

Learns from user command patterns to provide Warp-style intelligent suggestions.
"""

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openagent.core.history import HistoryManager


@dataclass
class CommandPattern:
    """A learned command pattern."""

    sequence: List[str]
    frequency: int
    success_rate: float
    contexts: List[str] = field(default_factory=list)
    avg_time_between: float = 0.0
    last_used: float = 0.0


@dataclass
class CommandPatterns:
    """Collection of learned command patterns."""

    sequences: List[CommandPattern] = field(default_factory=list)
    corrections: Dict[str, str] = field(default_factory=dict)
    context_commands: Dict[str, List[str]] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionContext:
    """Context for command prediction."""

    current_directory: str
    recent_commands: List[str] = field(default_factory=list)
    project_type: str = "unknown"
    git_branch: Optional[str] = None
    time_of_day: int = 0  # Hour of day
    day_of_week: int = 0  # Day of week


class HistoryIntelligence:
    """
    Advanced history analysis and command prediction.

    Learns from user patterns to provide intelligent suggestions
    similar to Warp AI's predictive capabilities.
    """

    def __init__(self, history_manager: Optional[HistoryManager] = None):
        """Initialize history intelligence.

        Args:
            history_manager: History manager instance
        """
        self.history_manager = history_manager or HistoryManager()
        self.patterns_cache: Optional[CommandPatterns] = None
        self.cache_expiry = 0
        self.cache_duration = 3600  # 1 hour

        # Learning parameters
        self.min_pattern_frequency = 2
        self.max_sequence_length = 5
        self.pattern_decay_factor = 0.95  # Older patterns get less weight

    async def analyze_command_patterns(self, limit: int = 1000) -> CommandPatterns:
        """
        Analyze command history to identify patterns.

        Args:
            limit: Maximum number of recent commands to analyze

        Returns:
            CommandPatterns with learned sequences and preferences
        """
        # Check cache
        if self.patterns_cache and time.time() < self.cache_expiry:
            return self.patterns_cache

        # Get command history
        history_blocks = self.history_manager.list_blocks(limit=limit)

        # Extract command sequences
        commands = []
        timestamps = []
        contexts = []

        for block in history_blocks:
            if block.get("input"):
                commands.append(block["input"])
                timestamps.append(block.get("timestamp", time.time()))
                contexts.append(
                    {
                        "cwd": block.get("context", {}).get("cwd", ""),
                        "project_type": block.get("context", {}).get(
                            "project_type", "unknown"
                        ),
                    }
                )

        # Analyze patterns
        patterns = await self._extract_command_sequences(commands, timestamps, contexts)
        corrections = await self._extract_corrections(history_blocks)
        context_commands = await self._analyze_context_commands(commands, contexts)
        preferences = await self._analyze_user_preferences(commands, contexts)

        # Create command patterns object
        command_patterns = CommandPatterns(
            sequences=patterns,
            corrections=corrections,
            context_commands=context_commands,
            user_preferences=preferences,
        )

        # Cache results
        self.patterns_cache = command_patterns
        self.cache_expiry = time.time() + self.cache_duration

        return command_patterns

    async def predict_next_command(self, context: PredictionContext) -> List[str]:
        """
        Predict likely next commands based on context and history.

        Args:
            context: Current prediction context

        Returns:
            List of predicted commands ordered by likelihood
        """
        patterns = await self.analyze_command_patterns()
        predictions = []

        # Get recent command sequence
        recent_sequence = context.recent_commands[-self.max_sequence_length :]

        # Find matching patterns
        matching_patterns = []
        for pattern in patterns.sequences:
            if self._sequence_matches(recent_sequence, pattern.sequence[:-1]):
                # This pattern matches our recent commands
                matching_patterns.append(
                    (
                        pattern.sequence[-1],  # Next command in pattern
                        pattern.frequency
                        * pattern.success_rate
                        * self._calculate_recency_weight(pattern.last_used),
                    )
                )

        # Sort by score
        matching_patterns.sort(key=lambda x: x[1], reverse=True)
        predictions.extend([cmd for cmd, score in matching_patterns[:5]])

        # Add context-specific commands
        if context.project_type in patterns.context_commands:
            context_cmds = patterns.context_commands[context.project_type]
            predictions.extend(context_cmds[:3])

        # Add directory-specific commands
        if context.current_directory in patterns.context_commands:
            dir_cmds = patterns.context_commands[context.current_directory]
            predictions.extend(dir_cmds[:2])

        # Remove duplicates while preserving order
        unique_predictions = []
        seen = set()
        for cmd in predictions:
            if cmd not in seen:
                unique_predictions.append(cmd)
                seen.add(cmd)

        return unique_predictions[:10]

    async def learn_from_correction(self, original: str, corrected: str):
        """
        Learn from user corrections to improve suggestions.

        Args:
            original: Original command that was suggested
            corrected: User's corrected version
        """
        # Store correction pattern
        patterns = await self.analyze_command_patterns()
        patterns.corrections[original] = corrected

        # Save to persistent storage
        await self._save_correction(original, corrected)

        # Invalidate cache to trigger reanalysis
        self.patterns_cache = None

    async def get_command_suggestions(
        self, partial: str, context: PredictionContext
    ) -> List[str]:
        """
        Get command suggestions for partial input.

        Args:
            partial: Partial command input
            context: Current context

        Returns:
            List of command suggestions
        """
        patterns = await self.analyze_command_patterns()
        suggestions = []

        # Check for corrections first
        if partial in patterns.corrections:
            suggestions.append(patterns.corrections[partial])

        # Find commands that start with partial
        all_commands = []
        for pattern in patterns.sequences:
            all_commands.extend(pattern.sequence)

        # Add context-specific commands
        if context.project_type in patterns.context_commands:
            all_commands.extend(patterns.context_commands[context.project_type])

        # Filter and rank suggestions
        matching_commands = [
            cmd for cmd in all_commands if cmd.startswith(partial) and cmd != partial
        ]

        # Count frequency and sort
        command_counts = Counter(matching_commands)
        suggestions.extend([cmd for cmd, count in command_counts.most_common(10)])

        return suggestions

    async def _extract_command_sequences(
        self,
        commands: List[str],
        timestamps: List[float],
        contexts: List[Dict[str, Any]],
    ) -> List[CommandPattern]:
        """Extract command sequences from history."""
        sequences = defaultdict(
            lambda: {"count": 0, "success": 0, "contexts": set(), "times": []}
        )

        # Extract sequences of various lengths
        for seq_len in range(2, self.max_sequence_length + 1):
            for i in range(len(commands) - seq_len + 1):
                sequence = tuple(commands[i : i + seq_len])
                key = " -> ".join(sequence)

                sequences[key]["count"] += 1
                sequences[key]["success"] += 1  # Assume success for now
                sequences[key]["contexts"].add(
                    contexts[i].get("project_type", "unknown")
                )
                sequences[key]["times"].append(timestamps[i])

        # Convert to CommandPattern objects
        patterns = []
        for seq_str, data in sequences.items():
            if data["count"] >= self.min_pattern_frequency:
                sequence = seq_str.split(" -> ")

                # Calculate time between commands
                times = sorted(data["times"])
                avg_time_between = 0.0
                if len(times) > 1:
                    time_diffs = [
                        times[i + 1] - times[i] for i in range(len(times) - 1)
                    ]
                    avg_time_between = sum(time_diffs) / len(time_diffs)

                pattern = CommandPattern(
                    sequence=sequence,
                    frequency=data["count"],
                    success_rate=data["success"] / data["count"],
                    contexts=list(data["contexts"]),
                    avg_time_between=avg_time_between,
                    last_used=max(times) if times else 0.0,
                )
                patterns.append(pattern)

        # Sort by frequency and recency
        patterns.sort(
            key=lambda p: p.frequency * self._calculate_recency_weight(p.last_used),
            reverse=True,
        )

        return patterns[:50]  # Limit to top patterns

    async def _extract_corrections(
        self, history_blocks: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Extract command corrections from history."""
        corrections = {}

        # Look for patterns where user modified a suggested command
        for i in range(len(history_blocks) - 1):
            current = history_blocks[i]
            next_block = history_blocks[i + 1]

            # Check if the next command is a modification of the current
            if current.get("metadata", {}).get("suggested") and next_block.get("input"):

                original = current.get("input", "")
                corrected = next_block.get("input", "")

                # Simple heuristic: if commands are similar but different
                if original != corrected and self._commands_similar(
                    original, corrected
                ):
                    corrections[original] = corrected

        return corrections

    async def _analyze_context_commands(
        self, commands: List[str], contexts: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Analyze which commands are used in which contexts."""
        context_commands = defaultdict(Counter)

        for command, context in zip(commands, contexts):
            # Group by project type
            project_type = context.get("project_type", "unknown")
            context_commands[project_type][command] += 1

            # Group by directory
            cwd = context.get("cwd", "")
            if cwd:
                context_commands[cwd][command] += 1

        # Convert to lists of top commands
        result = {}
        for context_type, command_counts in context_commands.items():
            result[context_type] = [
                cmd for cmd, count in command_counts.most_common(10)
            ]

        return result

    async def _analyze_user_preferences(
        self, commands: List[str], contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user preferences from command history."""
        preferences = {}

        # Analyze preferred tools
        tool_usage = Counter()
        for command in commands:
            if command.startswith("git "):
                tool_usage["git"] += 1
            elif command.startswith("docker "):
                tool_usage["docker"] += 1
            elif command.startswith("python "):
                tool_usage["python"] += 1
            elif command.startswith("npm ") or command.startswith("yarn "):
                tool_usage["node"] += 1
            elif command.startswith("cargo "):
                tool_usage["rust"] += 1

        preferences["preferred_tools"] = dict(tool_usage.most_common(5))

        # Analyze command complexity preference
        avg_command_length = (
            sum(len(cmd.split()) for cmd in commands) / len(commands) if commands else 0
        )
        preferences["complexity_preference"] = (
            "high" if avg_command_length > 4 else "low"
        )

        # Analyze time patterns
        current_time = time.time()
        recent_commands = [cmd for cmd in commands[-50:]]  # Recent commands
        preferences["recent_activity"] = len(recent_commands)

        return preferences

    def _sequence_matches(self, recent: List[str], pattern: List[str]) -> bool:
        """Check if recent commands match a pattern sequence."""
        if len(recent) < len(pattern):
            return False

        # Check if the end of recent matches the pattern
        recent_end = recent[-len(pattern) :]
        return recent_end == pattern

    def _commands_similar(self, cmd1: str, cmd2: str) -> bool:
        """Check if two commands are similar (for correction detection)."""
        # Simple similarity check
        words1 = set(cmd1.split())
        words2 = set(cmd2.split())

        # If they share at least 50% of words, consider them similar
        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union >= 0.5

    def _calculate_recency_weight(self, timestamp: float) -> float:
        """Calculate weight based on how recent a pattern was used."""
        days_ago = (time.time() - timestamp) / 86400  # Days since last use
        return self.pattern_decay_factor**days_ago

    async def _save_correction(self, original: str, corrected: str):
        """Save correction to persistent storage."""
        corrections_file = Path.home() / ".config" / "openagent" / "corrections.json"
        corrections_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if corrections_file.exists():
                corrections = json.loads(corrections_file.read_text())
            else:
                corrections = {}

            corrections[original] = {
                "corrected": corrected,
                "timestamp": time.time(),
                "count": corrections.get(original, {}).get("count", 0) + 1,
            }

            corrections_file.write_text(json.dumps(corrections, indent=2))
        except Exception:
            # Fail silently for corrections
            pass

    def get_command_insights(self, command: str) -> Dict[str, Any]:
        """Get insights about a specific command."""
        if not self.patterns_cache:
            return {}

        insights = {
            "frequency": 0,
            "success_rate": 0.0,
            "typical_next_commands": [],
            "common_contexts": [],
            "estimated_duration": 0.0,
        }

        # Find patterns containing this command
        for pattern in self.patterns_cache.sequences:
            if command in pattern.sequence:
                insights["frequency"] += pattern.frequency
                insights["success_rate"] = max(
                    insights["success_rate"], pattern.success_rate
                )
                insights["common_contexts"].extend(pattern.contexts)

                # Find what typically comes after this command
                try:
                    cmd_index = pattern.sequence.index(command)
                    if cmd_index < len(pattern.sequence) - 1:
                        next_cmd = pattern.sequence[cmd_index + 1]
                        insights["typical_next_commands"].append(next_cmd)
                except ValueError:
                    pass

        # Remove duplicates and get top items
        insights["typical_next_commands"] = list(
            set(insights["typical_next_commands"])
        )[:5]
        insights["common_contexts"] = list(set(insights["common_contexts"]))[:3]

        return insights

    async def predict_command_outcome(
        self, command: str, context: PredictionContext
    ) -> Dict[str, Any]:
        """
        Predict likely outcome of a command based on history.

        Args:
            command: Command to predict outcome for
            context: Current context

        Returns:
            Prediction with success probability and expected results
        """
        patterns = await self.analyze_command_patterns()

        # Find similar commands in history
        similar_commands = []
        for pattern in patterns.sequences:
            for cmd in pattern.sequence:
                if self._commands_similar(cmd, command):
                    similar_commands.append(
                        (cmd, pattern.success_rate, pattern.frequency)
                    )

        if not similar_commands:
            return {
                "success_probability": 0.8,  # Default assumption
                "confidence": "low",
                "expected_output": "Unknown",
                "estimated_duration": 1.0,
            }

        # Calculate weighted average success rate
        total_weight = sum(freq for _, _, freq in similar_commands)
        weighted_success = (
            sum(success * freq for _, success, freq in similar_commands) / total_weight
            if total_weight > 0
            else 0.8
        )

        return {
            "success_probability": weighted_success,
            "confidence": "high" if len(similar_commands) >= 3 else "medium",
            "expected_output": "Command output",  # Could be enhanced with actual output prediction
            "estimated_duration": 2.0,  # Could be predicted from history
        }

    def invalidate_cache(self):
        """Invalidate the patterns cache."""
        self.patterns_cache = None
        self.cache_expiry = 0
