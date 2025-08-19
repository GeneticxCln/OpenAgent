"""
Enhanced Context Engine v2 for OpenAgent.

Provides Warp-level context understanding including:
- Deep workspace analysis
- Project type detection
- Command history intelligence
- Environment awareness
- Git state understanding
"""

from .environment_detector import EnvironmentContext, EnvironmentDetector
from .history_intelligence import CommandPatterns, HistoryIntelligence
from .project_analyzer import ProjectContextEngine, ProjectType, WorkspaceContext

__all__ = [
    "ProjectContextEngine",
    "ProjectType",
    "WorkspaceContext",
    "HistoryIntelligence",
    "CommandPatterns",
    "EnvironmentDetector",
    "EnvironmentContext",
]
