"""
Debugging Assistant for OpenAgent.

Provides intelligent debugging support including stack trace analysis,
error diagnosis, and automated debugging script generation.
"""

import json
import re
import subprocess
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openagent.core.code_intelligence import CodeAnalysisEngine, CodeLanguage


class ErrorType(Enum):
    """Types of errors that can be diagnosed."""
    
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGICAL_ERROR = "logical_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    ATTRIBUTE_ERROR = "attribute_error"
    INDEX_ERROR = "index_error"
    KEY_ERROR = "key_error"
    FILE_ERROR = "file_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN_ERROR = "unknown_error"


class DebugSeverity(Enum):
    """Severity levels for debugging issues."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StackFrame:
    """Represents a single frame in a stack trace."""
    
    file_path: str
    function_name: str
    line_number: int
    code_line: Optional[str] = None
    local_vars: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebugAnalysis:
    """Results of debugging analysis."""
    
    error_type: ErrorType
    error_message: str
    severity: DebugSeverity
    
    # Stack trace information
    stack_frames: List[StackFrame] = field(default_factory=list)
    root_cause_frame: Optional[StackFrame] = None
    
    # Analysis results
    probable_causes: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    debug_commands: List[str] = field(default_factory=list)
    
    # Additional context
    related_files: List[str] = field(default_factory=list)
    environment_info: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence and metadata
    confidence: float = 0.0
    analysis_time: float = 0.0


@dataclass
class DebugSuggestion:
    """A debugging suggestion with context."""
    
    title: str
    description: str
    commands: List[str]
    explanation: str
    confidence: float
    category: str  # "inspection", "modification", "testing", etc.
    requires_confirmation: bool = False


class DebuggingAssistant:
    """
    Intelligent debugging assistant for various programming languages.
    
    Features:
    - Stack trace analysis and parsing
    - Error type classification and diagnosis
    - Context-aware debugging suggestions
    - Automated debug script generation
    - Multi-language support
    - Integration with code analysis
    """
    
    def __init__(self, code_analyzer: Optional[CodeAnalysisEngine] = None):
        """Initialize the debugging assistant."""
        self.code_analyzer = code_analyzer or CodeAnalysisEngine()
        
        # Error pattern database
        self._init_error_patterns()
        
        # Language-specific debugging commands
        self._init_debug_commands()
    
    def _init_error_patterns(self):
        """Initialize error pattern recognition."""
        self.error_patterns = {
            ErrorType.SYNTAX_ERROR: [
                r"SyntaxError",
                r"invalid syntax",
                r"unexpected token",
                r"missing \)",
                r"unexpected indent"
            ],
            ErrorType.IMPORT_ERROR: [
                r"ImportError",
                r"ModuleNotFoundError",
                r"No module named",
                r"cannot import name"
            ],
            ErrorType.TYPE_ERROR: [
                r"TypeError",
                r"unsupported operand type",
                r"not callable",
                r"takes .+ positional argument"
            ],
            ErrorType.ATTRIBUTE_ERROR: [
                r"AttributeError",
                r"has no attribute",
                r"'NoneType' object"
            ],
            ErrorType.INDEX_ERROR: [
                r"IndexError",
                r"list index out of range",
                r"string index out of range"
            ],
            ErrorType.KEY_ERROR: [
                r"KeyError",
                r"key .+ not found"
            ],
            ErrorType.FILE_ERROR: [
                r"FileNotFoundError",
                r"PermissionError",
                r"IsADirectoryError",
                r"No such file or directory"
            ],
            ErrorType.NETWORK_ERROR: [
                r"ConnectionError",
                r"TimeoutError",
                r"URLError",
                r"HTTPError"
            ]
        }
        
        # Common solutions for each error type
        self.error_solutions = {
            ErrorType.SYNTAX_ERROR: [
                "Check for missing parentheses, brackets, or quotes",
                "Verify proper indentation (Python)",
                "Look for typos in keywords and operators",
                "Check for missing colons after if/for/while statements"
            ],
            ErrorType.IMPORT_ERROR: [
                "Verify the module is installed: pip list",
                "Check the module name spelling",
                "Ensure the module is in your Python path",
                "Install missing packages: pip install <module_name>"
            ],
            ErrorType.TYPE_ERROR: [
                "Check variable types before operations",
                "Verify function arguments match expected types",
                "Use type conversion if needed (int(), str(), float())",
                "Check if variable is None before using it"
            ],
            ErrorType.ATTRIBUTE_ERROR: [
                "Verify the object has the expected attribute",
                "Check if variable is None",
                "Use hasattr() to check attribute existence",
                "Review object initialization"
            ],
            ErrorType.INDEX_ERROR: [
                "Check list/string length before accessing",
                "Use try-except for index access",
                "Verify loop bounds",
                "Use negative indexing carefully"
            ],
            ErrorType.KEY_ERROR: [
                "Check if key exists in dictionary",
                "Use dict.get() with default values",
                "Verify key spelling and type",
                "Use 'in' operator to check key existence"
            ]
        }
    
    def _init_debug_commands(self):
        """Initialize language-specific debugging commands."""
        self.debug_commands = {
            CodeLanguage.PYTHON: {
                "inspect_variable": "print(repr({var}))",
                "check_type": "print(type({var}))",
                "list_attributes": "print(dir({var}))",
                "check_none": "print('{var} is None:', {var} is None)",
                "trace_function": "import pdb; pdb.set_trace()",
                "print_locals": "print(locals())",
                "print_globals": "print(globals())",
                "check_module": "import sys; print('{module}' in sys.modules)",
                "list_methods": "print([method for method in dir({obj}) if callable(getattr({obj}, method))])"
            },
            CodeLanguage.JAVASCRIPT: {
                "inspect_variable": "console.log('{var}:', {var})",
                "check_type": "console.log('Type of {var}:', typeof {var})",
                "list_properties": "console.log('Properties:', Object.keys({var}))",
                "check_undefined": "console.log('{var} is undefined:', {var} === undefined)",
                "trace_function": "debugger;",
                "stack_trace": "console.trace()",
                "inspect_prototype": "console.log('Prototype:', Object.getPrototypeOf({obj}))"
            },
            CodeLanguage.GO: {
                "inspect_variable": "fmt.Printf(\"{var}: %+v\\n\", {var})",
                "check_type": "fmt.Printf(\"Type: %T\\n\", {var})",
                "check_nil": "fmt.Printf(\"{var} is nil: %t\\n\", {var} == nil)",
                "stack_trace": "debug.PrintStack()"
            },
            CodeLanguage.RUST: {
                "inspect_variable": "println!(\"{var}: {{:?}}\", {var});",
                "check_type": "println!(\"Type: {{:?}}\", std::any::type_name::<{type}>());",
                "debug_print": "dbg!({var});",
                "stack_trace": "println!(\"{{:?}}\", std::backtrace::Backtrace::capture());"
            }
        }
    
    def analyze_stack_trace(self, trace_text: str, language: CodeLanguage = CodeLanguage.PYTHON) -> DebugAnalysis:
        """
        Analyze a stack trace and provide debugging insights.
        
        Args:
            trace_text: Raw stack trace text
            language: Programming language of the stack trace
            
        Returns:
            DebugAnalysis with detailed information and suggestions
        """
        if language == CodeLanguage.PYTHON:
            return self._analyze_python_stack_trace(trace_text)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._analyze_javascript_stack_trace(trace_text)
        else:
            return self._analyze_generic_stack_trace(trace_text, language)
    
    def _analyze_python_stack_trace(self, trace_text: str) -> DebugAnalysis:
        """Analyze Python stack trace."""
        analysis = DebugAnalysis(
            error_type=ErrorType.UNKNOWN_ERROR,
            error_message="",
            severity=DebugSeverity.MEDIUM
        )
        
        # Extract error type and message
        error_match = re.search(r'(\w+Error): (.+)$', trace_text, re.MULTILINE)
        if error_match:
            error_name = error_match.group(1)
            analysis.error_message = error_match.group(2).strip()
            analysis.error_type = self._classify_error_type(error_name)
        
        # Parse stack frames
        frame_pattern = r'File "([^"]+)", line (\d+), in (.+)\n\s*(.+)'
        frames = []
        
        for match in re.finditer(frame_pattern, trace_text):
            frame = StackFrame(
                file_path=match.group(1),
                function_name=match.group(3),
                line_number=int(match.group(2)),
                code_line=match.group(4).strip()
            )
            frames.append(frame)
        
        analysis.stack_frames = frames
        
        # Identify root cause (usually the last user code frame)
        user_frames = [f for f in frames if not self._is_library_file(f.file_path)]
        if user_frames:
            analysis.root_cause_frame = user_frames[-1]
        elif frames:
            analysis.root_cause_frame = frames[-1]
        
        # Generate probable causes and fixes
        analysis.probable_causes = self._generate_probable_causes(analysis)
        analysis.suggested_fixes = self._generate_suggested_fixes(analysis)
        analysis.debug_commands = self._generate_debug_commands(analysis, CodeLanguage.PYTHON)
        
        # Calculate confidence
        analysis.confidence = self._calculate_confidence(analysis)
        
        return analysis
    
    def _analyze_javascript_stack_trace(self, trace_text: str) -> DebugAnalysis:
        """Analyze JavaScript stack trace."""
        analysis = DebugAnalysis(
            error_type=ErrorType.UNKNOWN_ERROR,
            error_message="",
            severity=DebugSeverity.MEDIUM
        )
        
        # Extract error message (first line usually)
        lines = trace_text.strip().split('\n')
        if lines:
            analysis.error_message = lines[0]
            analysis.error_type = self._classify_error_type(lines[0])
        
        # Parse stack frames
        frame_pattern = r'at (.+) \(([^:]+):(\d+):(\d+)\)'
        frames = []
        
        for match in re.finditer(frame_pattern, trace_text):
            frame = StackFrame(
                file_path=match.group(2),
                function_name=match.group(1),
                line_number=int(match.group(3))
            )
            frames.append(frame)
        
        analysis.stack_frames = frames
        if frames:
            analysis.root_cause_frame = frames[0]  # First frame is usually the issue
        
        analysis.probable_causes = self._generate_probable_causes(analysis)
        analysis.suggested_fixes = self._generate_suggested_fixes(analysis)
        analysis.debug_commands = self._generate_debug_commands(analysis, CodeLanguage.JAVASCRIPT)
        analysis.confidence = self._calculate_confidence(analysis)
        
        return analysis
    
    def _analyze_generic_stack_trace(self, trace_text: str, language: CodeLanguage) -> DebugAnalysis:
        """Analyze stack trace for other languages."""
        analysis = DebugAnalysis(
            error_type=ErrorType.RUNTIME_ERROR,
            error_message=trace_text[:200],  # First 200 chars
            severity=DebugSeverity.MEDIUM
        )
        
        # Basic error classification
        analysis.error_type = self._classify_error_type(trace_text)
        analysis.probable_causes = ["Runtime error occurred", "Check application logs"]
        analysis.suggested_fixes = ["Review recent code changes", "Check input data"]
        analysis.confidence = 0.3
        
        return analysis
    
    def suggest_debug_commands(self, analysis: DebugAnalysis, language: CodeLanguage) -> List[DebugSuggestion]:
        """Generate debugging command suggestions."""
        suggestions = []
        
        if not analysis.root_cause_frame:
            return suggestions
        
        frame = analysis.root_cause_frame
        
        # Variable inspection suggestions
        if analysis.error_type == ErrorType.ATTRIBUTE_ERROR:
            suggestions.append(DebugSuggestion(
                title="Inspect object attributes",
                description="Check what attributes the object actually has",
                commands=[
                    self._format_debug_command("list_attributes", language, var="<object_name>"),
                    self._format_debug_command("check_type", language, var="<object_name>"),
                    self._format_debug_command("check_none", language, var="<object_name>")
                ],
                explanation="This will help you understand what methods and attributes are available",
                confidence=0.8,
                category="inspection"
            ))
        
        elif analysis.error_type == ErrorType.TYPE_ERROR:
            suggestions.append(DebugSuggestion(
                title="Check variable types",
                description="Verify the types of variables involved in the operation",
                commands=[
                    self._format_debug_command("check_type", language, var="<variable_name>"),
                    self._format_debug_command("inspect_variable", language, var="<variable_name>")
                ],
                explanation="Type errors often occur when operations are performed on incompatible types",
                confidence=0.9,
                category="inspection"
            ))
        
        elif analysis.error_type == ErrorType.INDEX_ERROR:
            suggestions.append(DebugSuggestion(
                title="Check collection bounds",
                description="Verify the size and contents of the collection",
                commands=[
                    "print(f'Length: {len(<collection_name>)}')",
                    "print(f'Contents: {<collection_name>}')",
                    "print(f'Index being accessed: {<index_variable>}')"
                ],
                explanation="Index errors occur when accessing beyond collection boundaries",
                confidence=0.85,
                category="inspection"
            ))
        
        # General debugging suggestions
        suggestions.append(DebugSuggestion(
            title="Add breakpoint for detailed inspection",
            description="Set a breakpoint to examine the program state",
            commands=[self._format_debug_command("trace_function", language)],
            explanation="This will pause execution so you can inspect variables and call stack",
            confidence=0.7,
            category="debugging",
            requires_confirmation=True
        ))
        
        # File-specific suggestions
        if frame.file_path and Path(frame.file_path).exists():
            suggestions.append(DebugSuggestion(
                title="Review problematic code section",
                description=f"Examine the code around line {frame.line_number}",
                commands=[
                    f"head -n {frame.line_number + 5} {frame.file_path} | tail -n 11",
                    f"grep -n -A 3 -B 3 'line_content' {frame.file_path}"
                ],
                explanation="Understanding the context around the error line can reveal the issue",
                confidence=0.6,
                category="inspection"
            ))
        
        return suggestions
    
    def generate_debug_script(self, analysis: DebugAnalysis, language: CodeLanguage) -> str:
        """Generate a debugging script based on analysis."""
        if language == CodeLanguage.PYTHON:
            return self._generate_python_debug_script(analysis)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._generate_javascript_debug_script(analysis)
        else:
            return self._generate_generic_debug_script(analysis, language)
    
    def _generate_python_debug_script(self, analysis: DebugAnalysis) -> str:
        """Generate Python debugging script."""
        script_lines = [
            "#!/usr/bin/env python3",
            '"""',
            "Auto-generated debugging script",
            f"Error: {analysis.error_message}",
            f"Type: {analysis.error_type.value}",
            '"""',
            "",
            "import sys",
            "import pdb",
            "import traceback",
            ""
        ]
        
        if analysis.root_cause_frame:
            script_lines.extend([
                f"# Debug commands for {analysis.root_cause_frame.file_path}",
                f"# Line {analysis.root_cause_frame.line_number}: {analysis.root_cause_frame.function_name}",
                ""
            ])
        
        # Add specific debugging based on error type
        if analysis.error_type == ErrorType.ATTRIBUTE_ERROR:
            script_lines.extend([
                "# Attribute Error debugging",
                "def debug_attribute_error(obj, attr_name):",
                "    print(f'Object type: {type(obj)}')",
                "    print(f'Available attributes: {dir(obj)}')",
                "    print(f'Looking for attribute: {attr_name}')",
                "    print(f'Has attribute: {hasattr(obj, attr_name)}')",
                ""
            ])
        
        elif analysis.error_type == ErrorType.TYPE_ERROR:
            script_lines.extend([
                "# Type Error debugging",
                "def debug_type_error(*args):",
                "    for i, arg in enumerate(args):",
                "        print(f'Argument {i}: {arg} (type: {type(arg)})')",
                ""
            ])
        
        script_lines.extend([
            "# General debugging utilities",
            "def inspect_locals():",
            "    frame = sys._getframe(1)",
            "    print('Local variables:')",
            "    for name, value in frame.f_locals.items():",
            "        print(f'  {name}: {value} ({type(value)})')",
            "",
            "def safe_print(obj, name='object'):",
            "    try:",
            "        print(f'{name}: {obj}')",
            "    except Exception as e:",
            "        print(f'Cannot print {name}: {e}')",
            "",
            "if __name__ == '__main__':",
            "    print('Debug script ready. Import and use the debugging functions.')",
            "    print('Available functions: inspect_locals(), safe_print()')"
        ])
        
        if analysis.error_type in [ErrorType.ATTRIBUTE_ERROR, ErrorType.TYPE_ERROR]:
            error_func = analysis.error_type.value.replace('_error', '_error')
            script_lines.append(f"    print('Specific function: debug_{error_func}()')")
        
        return '\n'.join(script_lines)
    
    def _generate_javascript_debug_script(self, analysis: DebugAnalysis) -> str:
        """Generate JavaScript debugging script."""
        script_lines = [
            "// Auto-generated debugging script",
            f"// Error: {analysis.error_message}",
            f"// Type: {analysis.error_type.value}",
            "",
            "// Debugging utilities",
            "function inspectObject(obj, name = 'object') {",
            "    console.log(`${name}:`, obj);",
            "    console.log(`Type: ${typeof obj}`);",
            "    console.log(`Constructor: ${obj?.constructor?.name}`);",
            "    if (obj && typeof obj === 'object') {",
            "        console.log(`Keys: ${Object.keys(obj)}`);",
            "    }",
            "}",
            "",
            "function safeLog(value, name = 'value') {",
            "    try {",
            "        console.log(`${name}:`, value);",
            "    } catch (error) {",
            "        console.log(`Cannot log ${name}:`, error.message);",
            "    }",
            "}",
            "",
            "function debugTrace() {",
            "    console.trace('Debug trace:');",
            "}",
            "",
            "// Export utilities for use",
            "if (typeof module !== 'undefined') {",
            "    module.exports = { inspectObject, safeLog, debugTrace };",
            "}",
            "",
            "console.log('Debug script loaded. Available functions:');",
            "console.log('- inspectObject(obj, name)');",
            "console.log('- safeLog(value, name)');",
            "console.log('- debugTrace()');",
        ]
        
        return '\n'.join(script_lines)
    
    def _generate_generic_debug_script(self, analysis: DebugAnalysis, language: CodeLanguage) -> str:
        """Generate generic debugging script."""
        return f"""# Debug script for {language.value}
# Error: {analysis.error_message}
# Type: {analysis.error_type.value}

# Add debugging commands specific to your language and environment
# Review the error message and stack trace for clues
# Check recent changes and input data

echo "Debug information:"
echo "Error: {analysis.error_message}"
echo "Language: {language.value}"
"""
    
    def _classify_error_type(self, error_text: str) -> ErrorType:
        """Classify error type from error text."""
        error_text_lower = error_text.lower()
        
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern.lower(), error_text_lower):
                    return error_type
        
        return ErrorType.UNKNOWN_ERROR
    
    def _generate_probable_causes(self, analysis: DebugAnalysis) -> List[str]:
        """Generate probable causes based on error analysis."""
        causes = []
        
        if analysis.error_type in self.error_solutions:
            # Use predefined causes for known error types
            base_causes = self.error_solutions[analysis.error_type][:2]
            causes.extend(base_causes)
        
        # Context-specific causes
        if analysis.root_cause_frame:
            frame = analysis.root_cause_frame
            
            # Add file-specific causes
            if 'test' in frame.file_path.lower():
                causes.append("Test setup or assertion issue")
            
            if frame.function_name == '__init__':
                causes.append("Object initialization problem")
            
            if frame.function_name.startswith('_'):
                causes.append("Private method access issue")
        
        return causes[:4]  # Limit to top 4 causes
    
    def _generate_suggested_fixes(self, analysis: DebugAnalysis) -> List[str]:
        """Generate suggested fixes based on error analysis."""
        fixes = []
        
        if analysis.error_type in self.error_solutions:
            fixes.extend(self.error_solutions[analysis.error_type])
        
        # Add general fixes
        fixes.extend([
            "Add error handling with try-except blocks",
            "Add input validation and type checking",
            "Review recent code changes",
            "Check documentation and examples"
        ])
        
        return fixes[:6]  # Limit to top 6 fixes
    
    def _generate_debug_commands(self, analysis: DebugAnalysis, language: CodeLanguage) -> List[str]:
        """Generate debug commands for the specific language."""
        commands = []
        
        if language not in self.debug_commands:
            return ["# No specific debug commands available for this language"]
        
        lang_commands = self.debug_commands[language]
        
        # Always include basic inspection
        commands.append(lang_commands.get("inspect_variable", "").format(var="<variable_name>"))
        commands.append(lang_commands.get("check_type", "").format(var="<variable_name>"))
        
        # Error-specific commands
        if analysis.error_type == ErrorType.ATTRIBUTE_ERROR and "list_attributes" in lang_commands:
            commands.append(lang_commands["list_attributes"].format(var="<object_name>"))
        
        if analysis.error_type == ErrorType.TYPE_ERROR and "check_none" in lang_commands:
            commands.append(lang_commands["check_none"].format(var="<variable_name>"))
        
        # Add trace command
        if "trace_function" in lang_commands:
            commands.append(lang_commands["trace_function"])
        
        return [cmd for cmd in commands if cmd]  # Remove empty commands
    
    def _format_debug_command(self, command_name: str, language: CodeLanguage, **kwargs) -> str:
        """Format a debug command with parameters."""
        if language not in self.debug_commands:
            return f"# {command_name} not available for {language.value}"
        
        command_template = self.debug_commands[language].get(command_name, "")
        if not command_template:
            return f"# {command_name} not available"
        
        try:
            return command_template.format(**kwargs)
        except KeyError:
            return command_template
    
    def _is_library_file(self, file_path: str) -> bool:
        """Check if file path is from a library (not user code)."""
        library_indicators = [
            '/site-packages/',
            '/dist-packages/',
            '/lib/python',
            '/usr/lib/',
            'node_modules/',
            '.venv/',
            'venv/'
        ]
        
        return any(indicator in file_path for indicator in library_indicators)
    
    def _calculate_confidence(self, analysis: DebugAnalysis) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.0
        
        # Base confidence from error classification
        if analysis.error_type != ErrorType.UNKNOWN_ERROR:
            confidence += 0.4
        
        # Confidence from stack trace availability
        if analysis.stack_frames:
            confidence += 0.3
        
        # Confidence from root cause identification
        if analysis.root_cause_frame:
            confidence += 0.2
        
        # Confidence from having specific suggestions
        if analysis.probable_causes:
            confidence += 0.1
        
        return min(confidence, 1.0)


# Factory function for easy creation
def create_debugging_assistant(code_analyzer: Optional[CodeAnalysisEngine] = None) -> DebuggingAssistant:
    """Create a debugging assistant with default configuration."""
    return DebuggingAssistant(code_analyzer)


# Example usage function
async def debug_error(
    error_text: str,
    language: CodeLanguage = CodeLanguage.PYTHON,
    assistant: Optional[DebuggingAssistant] = None
) -> Dict[str, Any]:
    """Debug an error and return comprehensive analysis."""
    if assistant is None:
        assistant = create_debugging_assistant()
    
    try:
        # Analyze the stack trace
        analysis = assistant.analyze_stack_trace(error_text, language)
        
        # Generate debugging suggestions
        suggestions = assistant.suggest_debug_commands(analysis, language)
        
        # Generate debug script
        debug_script = assistant.generate_debug_script(analysis, language)
        
        return {
            'error_type': analysis.error_type.value,
            'error_message': analysis.error_message,
            'severity': analysis.severity.value,
            'confidence': analysis.confidence,
            'stack_frames': [
                {
                    'file': frame.file_path,
                    'function': frame.function_name,
                    'line': frame.line_number,
                    'code': frame.code_line
                } for frame in analysis.stack_frames
            ],
            'root_cause': {
                'file': analysis.root_cause_frame.file_path,
                'function': analysis.root_cause_frame.function_name,
                'line': analysis.root_cause_frame.line_number
            } if analysis.root_cause_frame else None,
            'probable_causes': analysis.probable_causes,
            'suggested_fixes': analysis.suggested_fixes,
            'debug_commands': analysis.debug_commands,
            'suggestions': [
                {
                    'title': s.title,
                    'description': s.description,
                    'commands': s.commands,
                    'explanation': s.explanation,
                    'confidence': s.confidence,
                    'category': s.category
                } for s in suggestions
            ],
            'debug_script': debug_script
        }
    
    except Exception as e:
        return {
            'error': f'Failed to analyze debug information: {str(e)}',
            'error_text': error_text[:500]  # First 500 chars
        }
