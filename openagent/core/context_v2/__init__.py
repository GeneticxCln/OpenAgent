"""
Enhanced Context Engine v2 for OpenAgent.

Provides Warp-level context understanding including:
- Deep workspace analysis
- Project type detection
- Command history intelligence
- Environment awareness
- Git state understanding
"""

from .project_analyzer import ProjectContextEngine, ProjectType, WorkspaceContext
from .history_intelligence import HistoryIntelligence, CommandPatterns
from .environment_detector import EnvironmentDetector, EnvironmentContext

__all__ = [
    "ProjectContextEngine",
    "ProjectType", 
    "WorkspaceContext",
    "HistoryIntelligence",
    "CommandPatterns",
    "EnvironmentDetector",
    "EnvironmentContext",
]
