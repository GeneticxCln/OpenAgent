"""
Code Intelligence System for OpenAgent.

Provides advanced code analysis, pattern detection, and intelligent code understanding
similar to Warp's code intelligence features.
"""

import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import esprima  # For JavaScript/TypeScript parsing
except ImportError:
    esprima = None


class CodeLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    BASH = "bash"
    SQL = "sql"
    JSON = "json"
    YAML = "yaml"
    DOCKERFILE = "dockerfile"
    UNKNOWN = "unknown"


class PatternType(Enum):
    """Types of code patterns that can be detected."""

    DESIGN_PATTERN = "design_pattern"
    ANTI_PATTERN = "anti_pattern"
    CODE_SMELL = "code_smell"
    SECURITY_ISSUE = "security_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    STYLE_VIOLATION = "style_violation"
    COMPLEXITY_ISSUE = "complexity_issue"


class ImprovementType(Enum):
    """Types of code improvements."""

    PERFORMANCE = "performance"
    SECURITY = "security"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class CodeStructure:
    """Represents the structure of analyzed code."""

    language: CodeLanguage
    file_path: Optional[Path] = None

    # AST representation
    ast_nodes: List[Dict[str, Any]] = field(default_factory=list)

    # Functions and methods
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)

    # Imports and dependencies
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Metrics
    lines_of_code: int = 0
    complexity_score: int = 0
    maintainability_index: float = 0.0

    # Metadata
    docstring: Optional[str] = None
    comments: List[str] = field(default_factory=list)

    # Issues found
    syntax_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CodePattern:
    """A detected code pattern."""

    pattern_type: PatternType
    name: str
    description: str
    location: Dict[str, Any]  # Line numbers, function names, etc.
    confidence: float
    severity: str  # low, medium, high, critical

    # Context
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None
    references: List[str] = field(default_factory=list)


@dataclass
class CodeImprovement:
    """A suggested code improvement."""

    improvement_type: ImprovementType
    title: str
    description: str
    location: Dict[str, Any]
    confidence: float
    impact: str  # low, medium, high

    # Implementation details
    before_code: Optional[str] = None
    after_code: Optional[str] = None
    diff: Optional[str] = None

    # Additional info
    rationale: str = ""
    references: List[str] = field(default_factory=list)
    automated: bool = False  # Can this improvement be automated?


class CodeAnalysisEngine:
    """
    Advanced code analysis engine for understanding code structure and quality.

    Features:
    - Multi-language AST parsing and analysis
    - Design pattern and anti-pattern detection
    - Code quality metrics and scoring
    - Performance and security analysis
    - Automated improvement suggestions
    """

    def __init__(self):
        """Initialize the code analysis engine."""
        self.supported_languages = {
            ".py": CodeLanguage.PYTHON,
            ".js": CodeLanguage.JAVASCRIPT,
            ".jsx": CodeLanguage.JAVASCRIPT,
            ".ts": CodeLanguage.TYPESCRIPT,
            ".tsx": CodeLanguage.TYPESCRIPT,
            ".rs": CodeLanguage.RUST,
            ".go": CodeLanguage.GO,
            ".sh": CodeLanguage.BASH,
            ".bash": CodeLanguage.BASH,
            ".sql": CodeLanguage.SQL,
            ".json": CodeLanguage.JSON,
            ".yaml": CodeLanguage.YAML,
            ".yml": CodeLanguage.YAML,
            "Dockerfile": CodeLanguage.DOCKERFILE,
        }

        # Pattern definitions
        self._init_pattern_database()

    def _init_pattern_database(self):
        """Initialize the pattern detection database."""
        self.patterns = {
            CodeLanguage.PYTHON: {
                PatternType.DESIGN_PATTERN: [
                    {
                        "name": "Singleton Pattern",
                        "indicators": ["__new__", "instance", "_instance"],
                        "description": "Singleton design pattern implementation",
                    },
                    {
                        "name": "Factory Pattern",
                        "indicators": ["create_", "factory", "make_"],
                        "description": "Factory method or abstract factory pattern",
                    },
                    {
                        "name": "Observer Pattern",
                        "indicators": ["subscribe", "notify", "observer", "listener"],
                        "description": "Observer pattern implementation",
                    },
                ],
                PatternType.ANTI_PATTERN: [
                    {
                        "name": "God Class",
                        "indicators": ["too_many_methods", "high_complexity"],
                        "description": "Class with too many responsibilities",
                    },
                    {
                        "name": "Magic Numbers",
                        "indicators": ["hardcoded_numbers"],
                        "description": "Unexplained numeric literals in code",
                    },
                ],
                PatternType.CODE_SMELL: [
                    {
                        "name": "Long Method",
                        "indicators": ["method_too_long"],
                        "description": "Method with too many lines of code",
                    },
                    {
                        "name": "Duplicate Code",
                        "indicators": ["code_duplication"],
                        "description": "Similar code blocks that could be refactored",
                    },
                ],
                PatternType.SECURITY_ISSUE: [
                    {
                        "name": "SQL Injection Risk",
                        "indicators": ["format", "execute", "sql"],
                        "description": "Potential SQL injection vulnerability",
                    },
                    {
                        "name": "Hardcoded Secrets",
                        "indicators": ["password", "api_key", "secret"],
                        "description": "Hardcoded credentials or secrets",
                    },
                ],
            }
        }

    def detect_language(self, file_path: Path) -> CodeLanguage:
        """Auto-detect programming language from file path."""
        if file_path.name == "Dockerfile":
            return CodeLanguage.DOCKERFILE

        suffix = file_path.suffix.lower()
        return self.supported_languages.get(suffix, CodeLanguage.UNKNOWN)

    def parse_code_structure(
        self, code: str, language: CodeLanguage, file_path: Optional[Path] = None
    ) -> CodeStructure:
        """Parse code into structured representation."""
        if language == CodeLanguage.PYTHON:
            return self._parse_python_structure(code, file_path)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._parse_javascript_structure(code, file_path)
        elif language == CodeLanguage.TYPESCRIPT:
            return self._parse_typescript_structure(code, file_path)
        else:
            return self._parse_generic_structure(code, language, file_path)

    def _parse_python_structure(
        self, code: str, file_path: Optional[Path] = None
    ) -> CodeStructure:
        """Parse Python code structure using AST."""
        structure = CodeStructure(
            language=CodeLanguage.PYTHON,
            file_path=file_path,
            lines_of_code=len(code.split("\n")),
        )

        try:
            tree = ast.parse(code)

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node),
                        "decorators": [
                            self._ast_to_string(dec) for dec in node.decorator_list
                        ],
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "complexity": self._calculate_complexity(node),
                    }
                    structure.functions.append(func_info)

                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": getattr(node, "end_lineno", node.lineno),
                        "bases": [self._ast_to_string(base) for base in node.bases],
                        "docstring": ast.get_docstring(node),
                        "decorators": [
                            self._ast_to_string(dec) for dec in node.decorator_list
                        ],
                        "methods": [],
                        "complexity": 0,
                    }

                    # Get methods in the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                "name": item.name,
                                "line_start": item.lineno,
                                "args": [arg.arg for arg in item.args.args],
                                "is_property": any(
                                    self._ast_to_string(dec) == "property"
                                    for dec in item.decorator_list
                                ),
                                "is_static": any(
                                    self._ast_to_string(dec) == "staticmethod"
                                    for dec in item.decorator_list
                                ),
                                "is_class": any(
                                    self._ast_to_string(dec) == "classmethod"
                                    for dec in item.decorator_list
                                ),
                            }
                            class_info["methods"].append(method_info)
                            class_info["complexity"] += self._calculate_complexity(item)

                    structure.classes.append(class_info)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        import_name = alias.name
                        if isinstance(node, ast.ImportFrom) and node.module:
                            import_name = f"{node.module}.{import_name}"
                        structure.imports.append(import_name)

            # Calculate overall complexity
            structure.complexity_score = sum(
                func["complexity"] for func in structure.functions
            )
            structure.complexity_score += sum(
                cls["complexity"] for cls in structure.classes
            )

            # Extract module docstring
            if (
                isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)
            ):
                structure.docstring = tree.body[0].value.value

        except SyntaxError as e:
            structure.syntax_errors.append(str(e))
        except Exception as e:
            structure.warnings.append(f"Parse error: {str(e)}")

        return structure

    def _parse_javascript_structure(
        self, code: str, file_path: Optional[Path] = None
    ) -> CodeStructure:
        """Parse JavaScript code structure."""
        structure = CodeStructure(
            language=CodeLanguage.JAVASCRIPT,
            file_path=file_path,
            lines_of_code=len(code.split("\n")),
        )

        # Basic regex-based parsing for JavaScript (could be enhanced with esprima)
        # Extract functions
        func_pattern = r"function\s+(\w+)\s*\([^)]*\)"
        for match in re.finditer(func_pattern, code):
            func_info = {
                "name": match.group(1),
                "line_start": code[: match.start()].count("\n") + 1,
                "type": "function",
            }
            structure.functions.append(func_info)

        # Extract arrow functions
        arrow_pattern = r"const\s+(\w+)\s*=\s*\([^)]*\)\s*=>"
        for match in re.finditer(arrow_pattern, code):
            func_info = {
                "name": match.group(1),
                "line_start": code[: match.start()].count("\n") + 1,
                "type": "arrow_function",
            }
            structure.functions.append(func_info)

        # Extract classes
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(class_pattern, code):
            class_info = {
                "name": match.group(1),
                "line_start": code[: match.start()].count("\n") + 1,
                "extends": match.group(2) if match.group(2) else None,
            }
            structure.classes.append(class_info)

        # Extract imports
        import_patterns = [
            r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
            r'const\s+.*?\s+=\s+require\(["\']([^"\']+)["\']\)',
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, code):
                structure.imports.append(match.group(1))

        return structure

    def _parse_typescript_structure(
        self, code: str, file_path: Optional[Path] = None
    ) -> CodeStructure:
        """Parse TypeScript code structure."""
        # For now, use JavaScript parsing with some TypeScript enhancements
        structure = self._parse_javascript_structure(code, file_path)
        structure.language = CodeLanguage.TYPESCRIPT

        # Extract interfaces
        interface_pattern = r"interface\s+(\w+)"
        interfaces = []
        for match in re.finditer(interface_pattern, code):
            interface_info = {
                "name": match.group(1),
                "line_start": code[: match.start()].count("\n") + 1,
            }
            interfaces.append(interface_info)

        structure.ast_nodes.append({"type": "interfaces", "data": interfaces})

        # Extract type definitions
        type_pattern = r"type\s+(\w+)\s*="
        types = []
        for match in re.finditer(type_pattern, code):
            type_info = {
                "name": match.group(1),
                "line_start": code[: match.start()].count("\n") + 1,
            }
            types.append(type_info)

        structure.ast_nodes.append({"type": "types", "data": types})

        return structure

    def _parse_generic_structure(
        self, code: str, language: CodeLanguage, file_path: Optional[Path] = None
    ) -> CodeStructure:
        """Parse generic code structure for unsupported languages."""
        return CodeStructure(
            language=language, file_path=file_path, lines_of_code=len(code.split("\n"))
        )

    def detect_code_patterns(
        self, structure: CodeStructure, code: str
    ) -> List[CodePattern]:
        """Detect design patterns, anti-patterns, and code smells."""
        patterns = []

        if structure.language not in self.patterns:
            return patterns

        language_patterns = self.patterns[structure.language]

        for pattern_type, pattern_list in language_patterns.items():
            for pattern_def in pattern_list:
                detected = self._check_pattern(pattern_def, structure, code)
                if detected:
                    pattern = CodePattern(
                        pattern_type=pattern_type,
                        name=pattern_def["name"],
                        description=pattern_def["description"],
                        location=detected["location"],
                        confidence=detected["confidence"],
                        severity=detected.get("severity", "medium"),
                        code_snippet=detected.get("code_snippet"),
                        suggestion=detected.get("suggestion"),
                    )
                    patterns.append(pattern)

        return patterns

    def suggest_improvements(
        self, structure: CodeStructure, code: str, patterns: List[CodePattern] = None
    ) -> List[CodeImprovement]:
        """Suggest code improvements based on analysis."""
        improvements = []
        patterns = patterns or []

        # Performance improvements
        perf_improvements = self._suggest_performance_improvements(structure, code)
        improvements.extend(perf_improvements)

        # Security improvements
        sec_improvements = self._suggest_security_improvements(structure, code)
        improvements.extend(sec_improvements)

        # Readability improvements
        read_improvements = self._suggest_readability_improvements(structure, code)
        improvements.extend(read_improvements)

        # Pattern-based improvements
        pattern_improvements = self._suggest_pattern_improvements(patterns, code)
        improvements.extend(pattern_improvements)

        return improvements

    def generate_tests(
        self, function_info: Dict[str, Any], code: str, language: CodeLanguage
    ) -> Optional[str]:
        """Auto-generate unit tests for functions."""
        if language == CodeLanguage.PYTHON:
            return self._generate_python_tests(function_info, code)
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            return self._generate_javascript_tests(function_info, code)

        return None

    def _check_pattern(
        self, pattern_def: Dict, structure: CodeStructure, code: str
    ) -> Optional[Dict[str, Any]]:
        """Check if a specific pattern exists in the code."""
        indicators = pattern_def.get("indicators", [])

        # Simple keyword-based detection (can be made more sophisticated)
        code_lower = code.lower()
        matches = 0
        total_indicators = len(indicators)

        for indicator in indicators:
            if indicator in code_lower:
                matches += 1

        if matches > 0:
            confidence = matches / total_indicators
            return {
                "location": {
                    "line": 1
                },  # Would be more specific in real implementation
                "confidence": confidence,
                "severity": "medium" if confidence > 0.5 else "low",
            }

        return None

    def _suggest_performance_improvements(
        self, structure: CodeStructure, code: str
    ) -> List[CodeImprovement]:
        """Suggest performance-related improvements."""
        improvements = []

        # Example: Detect inefficient loops
        if "for" in code and "append" in code:
            improvement = CodeImprovement(
                improvement_type=ImprovementType.PERFORMANCE,
                title="Consider using list comprehension",
                description="List comprehensions are typically faster than loops with append()",
                location={"type": "general"},
                confidence=0.7,
                impact="medium",
                rationale="List comprehensions are optimized in Python and generally perform better",
                automated=True,
            )
            improvements.append(improvement)

        return improvements

    def _suggest_security_improvements(
        self, structure: CodeStructure, code: str
    ) -> List[CodeImprovement]:
        """Suggest security-related improvements."""
        improvements = []

        # Check for potential SQL injection
        sql_patterns = ["execute(", "format(", "% "]
        if any(pattern in code for pattern in sql_patterns) and "sql" in code.lower():
            improvement = CodeImprovement(
                improvement_type=ImprovementType.SECURITY,
                title="Potential SQL injection risk",
                description="Use parameterized queries to prevent SQL injection",
                location={"type": "general"},
                confidence=0.6,
                impact="high",
                rationale="String formatting in SQL queries can lead to injection vulnerabilities",
                references=["https://owasp.org/www-community/attacks/SQL_Injection"],
            )
            improvements.append(improvement)

        return improvements

    def _suggest_readability_improvements(
        self, structure: CodeStructure, code: str
    ) -> List[CodeImprovement]:
        """Suggest readability improvements."""
        improvements = []

        # Check for long functions
        for func in structure.functions:
            if func.get("complexity", 0) > 10:
                improvement = CodeImprovement(
                    improvement_type=ImprovementType.READABILITY,
                    title=f"Function '{func['name']}' is complex",
                    description="Consider breaking this function into smaller, more focused functions",
                    location={
                        "function": func["name"],
                        "line": func.get("line_start", 0),
                    },
                    confidence=0.8,
                    impact="medium",
                    rationale="Functions with high complexity are harder to understand and maintain",
                )
                improvements.append(improvement)

        return improvements

    def _suggest_pattern_improvements(
        self, patterns: List[CodePattern], code: str
    ) -> List[CodeImprovement]:
        """Suggest improvements based on detected patterns."""
        improvements = []

        for pattern in patterns:
            if pattern.pattern_type == PatternType.ANTI_PATTERN:
                improvement = CodeImprovement(
                    improvement_type=ImprovementType.MAINTAINABILITY,
                    title=f"Refactor {pattern.name}",
                    description=f"Consider refactoring this {pattern.name.lower()}",
                    location=pattern.location,
                    confidence=pattern.confidence,
                    impact="high" if pattern.severity == "high" else "medium",
                    rationale=pattern.description,
                )
                improvements.append(improvement)

        return improvements

    def _generate_python_tests(self, function_info: Dict[str, Any], code: str) -> str:
        """Generate Python unit tests for a function."""
        func_name = function_info["name"]
        args = function_info.get("args", [])

        # Basic test template
        test_template = f"""import pytest
from your_module import {func_name}


def test_{func_name}_basic():
    \"\"\"Test basic functionality of {func_name}.\"\"\"
    # TODO: Add test cases
    result = {func_name}({', '.join(['None' for _ in args])})
    assert result is not None


def test_{func_name}_edge_cases():
    \"\"\"Test edge cases for {func_name}.\"\"\"
    # TODO: Add edge case tests
    pass


def test_{func_name}_error_handling():
    \"\"\"Test error handling in {func_name}.\"\"\"
    # TODO: Add error case tests
    pass
"""

        return test_template

    def _generate_javascript_tests(
        self, function_info: Dict[str, Any], code: str
    ) -> str:
        """Generate JavaScript unit tests for a function."""
        func_name = function_info["name"]

        test_template = f"""const {{ {func_name} }} = require('./your-module');

describe('{func_name}', () => {{
    test('should work with basic input', () => {{
        // TODO: Add test cases
        const result = {func_name}();
        expect(result).toBeDefined();
    }});
    
    test('should handle edge cases', () => {{
        // TODO: Add edge case tests
    }});
    
    test('should handle errors gracefully', () => {{
        // TODO: Add error case tests
    }});
}});
"""

        return test_template

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1

        return complexity

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)


# Factory function for easy creation
def create_code_analysis_engine() -> CodeAnalysisEngine:
    """Create a code analysis engine with default configuration."""
    return CodeAnalysisEngine()


# Example usage function
async def analyze_file(
    file_path: Path, engine: Optional[CodeAnalysisEngine] = None
) -> Dict[str, Any]:
    """Analyze a code file and return comprehensive results."""
    if engine is None:
        engine = create_code_analysis_engine()

    try:
        code = file_path.read_text(encoding="utf-8")
        language = engine.detect_language(file_path)

        if language == CodeLanguage.UNKNOWN:
            return {
                "error": f"Unsupported file type: {file_path.suffix}",
                "language": language.value,
            }

        # Parse code structure
        structure = engine.parse_code_structure(code, language, file_path)

        # Detect patterns
        patterns = engine.detect_code_patterns(structure, code)

        # Suggest improvements
        improvements = engine.suggest_improvements(structure, code, patterns)

        return {
            "file_path": str(file_path),
            "language": language.value,
            "structure": {
                "lines_of_code": structure.lines_of_code,
                "functions": len(structure.functions),
                "classes": len(structure.classes),
                "imports": len(structure.imports),
                "complexity_score": structure.complexity_score,
            },
            "functions": structure.functions,
            "classes": structure.classes,
            "patterns": [
                {
                    "type": p.pattern_type.value,
                    "name": p.name,
                    "description": p.description,
                    "confidence": p.confidence,
                    "severity": p.severity,
                }
                for p in patterns
            ],
            "improvements": [
                {
                    "type": i.improvement_type.value,
                    "title": i.title,
                    "description": i.description,
                    "confidence": i.confidence,
                    "impact": i.impact,
                    "automated": i.automated,
                }
                for i in improvements
            ],
            "syntax_errors": structure.syntax_errors,
            "warnings": structure.warnings,
        }

    except Exception as e:
        return {
            "error": f"Failed to analyze file: {str(e)}",
            "file_path": str(file_path),
        }
