"""
Code Generator System for OpenAgent.

Provides intelligent code generation, refactoring, and automated code improvements.
Integrates with the CodeAnalysisEngine for context-aware generation.
"""

import ast
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from openagent.core.code_intelligence import (
    CodeAnalysisEngine,
    CodeImprovement,
    CodeLanguage,
    CodeStructure,
    ImprovementType,
    create_code_analysis_engine,
)


class GenerationType(Enum):
    """Types of code generation."""

    FUNCTION = "function"
    CLASS = "class"
    TEST = "test"
    DOCSTRING = "docstring"
    REFACTOR = "refactor"
    BUG_FIX = "bug_fix"
    OPTIMIZATION = "optimization"
    BOILERPLATE = "boilerplate"


@dataclass
class GenerationRequest:
    """Request for code generation."""

    type: GenerationType
    description: str
    language: CodeLanguage
    context: Optional[CodeStructure] = None
    existing_code: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    style_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedCode:
    """Generated code result."""

    code: str
    language: CodeLanguage
    type: GenerationType
    confidence: float

    # Metadata
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    imports_needed: List[str] = field(default_factory=list)

    # Quality metrics
    estimated_complexity: int = 0
    test_coverage_suggestions: List[str] = field(default_factory=list)

    # Alternative implementations
    alternatives: List[str] = field(default_factory=list)


class CodeGenerator:
    """
    Intelligent code generation system.

    Features:
    - Context-aware code generation
    - Multiple language support
    - Style-consistent code generation
    - Integration with code analysis
    - Refactoring and optimization
    - Test generation
    """

    def __init__(self, analyzer: Optional[CodeAnalysisEngine] = None):
        """Initialize code generator."""
        self.analyzer = analyzer or create_code_analysis_engine()

        # Code templates for different languages
        self._init_templates()

        # Style preferences
        self.default_styles = {
            CodeLanguage.PYTHON: {
                "max_line_length": 88,
                "use_type_hints": True,
                "docstring_style": "google",  # google, numpy, sphinx
                "import_style": "absolute",
            },
            CodeLanguage.JAVASCRIPT: {
                "max_line_length": 100,
                "use_semicolons": True,
                "quote_style": "single",
                "async_style": "async_await",
            },
            CodeLanguage.TYPESCRIPT: {
                "max_line_length": 100,
                "use_semicolons": True,
                "quote_style": "single",
                "strict_types": True,
            },
        }

    def _init_templates(self):
        """Initialize code generation templates."""
        self.templates = {
            CodeLanguage.PYTHON: {
                GenerationType.FUNCTION: """def {name}({params}) -> {return_type}:
    \"\"\"
    {description}
    
    Args:
{args_doc}
    
    Returns:
        {return_doc}
    \"\"\"
    {body}""",
                GenerationType.CLASS: """class {name}({bases}):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self{init_params}):
        \"\"\"Initialize {name}.\"\"\"
{init_body}
{methods}""",
                GenerationType.TEST: """def test_{function_name}():
    \"\"\"Test {function_name} function.\"\"\"
    # Arrange
{arrange}
    
    # Act
    result = {function_name}({test_args})
    
    # Assert
{assertions}""",
            },
            CodeLanguage.JAVASCRIPT: {
                GenerationType.FUNCTION: """function {name}({params}) {{
    /**
     * {description}
{param_docs}
     * @returns {{{return_type}}} {return_doc}
     */
{body}
}}""",
                GenerationType.CLASS: """class {name}{extends} {{
    /**
     * {description}
     */
    constructor({constructor_params}) {{
{constructor_body}
    }}
{methods}
}}""",
                GenerationType.TEST: """describe('{function_name}', () => {{
    test('should {test_description}', () => {{
        // Arrange
{arrange}
        
        // Act
        const result = {function_name}({test_args});
        
        // Assert
{assertions}
    }});
}});""",
            },
        }

    async def generate_code(self, request: GenerationRequest) -> GeneratedCode:
        """
        Generate code based on the request.

        Args:
            request: Code generation request

        Returns:
            Generated code with metadata
        """
        if request.type == GenerationType.FUNCTION:
            return await self._generate_function(request)
        elif request.type == GenerationType.CLASS:
            return await self._generate_class(request)
        elif request.type == GenerationType.TEST:
            return await self._generate_test(request)
        elif request.type == GenerationType.DOCSTRING:
            return await self._generate_docstring(request)
        elif request.type == GenerationType.REFACTOR:
            return await self._refactor_code(request)
        elif request.type == GenerationType.BUG_FIX:
            return await self._generate_bug_fix(request)
        elif request.type == GenerationType.OPTIMIZATION:
            return await self._optimize_code(request)
        elif request.type == GenerationType.BOILERPLATE:
            return await self._generate_boilerplate(request)
        else:
            raise ValueError(f"Unsupported generation type: {request.type}")

    async def _generate_function(self, request: GenerationRequest) -> GeneratedCode:
        """Generate a function based on description."""
        # Parse parameters from description or request
        func_info = self._parse_function_description(request.description)

        # Get style preferences
        style = self._get_style_preferences(request.language, request.style_preferences)

        # Generate function body based on description
        body = self._generate_function_body(func_info, request.language, style)

        if request.language == CodeLanguage.PYTHON:
            code = self._generate_python_function(func_info, body, style)
        elif request.language == CodeLanguage.JAVASCRIPT:
            code = self._generate_javascript_function(func_info, body, style)
        else:
            code = self._generate_generic_function(func_info, body, request.language)

        return GeneratedCode(
            code=code,
            language=request.language,
            type=GenerationType.FUNCTION,
            confidence=0.8,
            description=f"Generated function: {func_info.get('name', 'unnamed')}",
            imports_needed=self._determine_imports(code, request.language),
            estimated_complexity=self._estimate_complexity(code),
            test_coverage_suggestions=[
                "Test with valid input",
                "Test with edge cases",
                "Test error handling",
            ],
        )

    async def _generate_class(self, request: GenerationRequest) -> GeneratedCode:
        """Generate a class based on description."""
        class_info = self._parse_class_description(request.description)

        style = self._get_style_preferences(request.language, request.style_preferences)

        if request.language == CodeLanguage.PYTHON:
            code = self._generate_python_class(class_info, style)
        elif request.language == CodeLanguage.JAVASCRIPT:
            code = self._generate_javascript_class(class_info, style)
        else:
            code = self._generate_generic_class(class_info, request.language)

        return GeneratedCode(
            code=code,
            language=request.language,
            type=GenerationType.CLASS,
            confidence=0.7,
            description=f"Generated class: {class_info.get('name', 'unnamed')}",
            imports_needed=self._determine_imports(code, request.language),
            estimated_complexity=self._estimate_complexity(code),
            test_coverage_suggestions=[
                "Test class instantiation",
                "Test public methods",
                "Test edge cases and error handling",
            ],
        )

    async def _generate_test(self, request: GenerationRequest) -> GeneratedCode:
        """Generate unit tests for existing code."""
        if not request.existing_code:
            raise ValueError("Existing code required for test generation")

        # Analyze the existing code
        structure = self.analyzer.parse_code_structure(
            request.existing_code, request.language
        )

        test_code = self._generate_test_cases(structure, request.language)

        return GeneratedCode(
            code=test_code,
            language=request.language,
            type=GenerationType.TEST,
            confidence=0.6,
            description="Generated unit tests",
            imports_needed=self._determine_test_imports(request.language),
            test_coverage_suggestions=[
                "Add more edge case tests",
                "Test error conditions",
                "Add integration tests",
            ],
        )

    async def _generate_docstring(self, request: GenerationRequest) -> GeneratedCode:
        """Generate documentation for existing code."""
        if not request.existing_code:
            raise ValueError("Existing code required for docstring generation")

        # Parse the code to understand structure
        if request.language == CodeLanguage.PYTHON:
            docstring = self._generate_python_docstring(request.existing_code)
        elif request.language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            docstring = self._generate_jsdoc(request.existing_code)
        else:
            docstring = self._generate_generic_documentation(request.existing_code)

        return GeneratedCode(
            code=docstring,
            language=request.language,
            type=GenerationType.DOCSTRING,
            confidence=0.8,
            description="Generated documentation",
            imports_needed=[],
        )

    async def _refactor_code(self, request: GenerationRequest) -> GeneratedCode:
        """Refactor existing code for better structure/performance."""
        if not request.existing_code:
            raise ValueError("Existing code required for refactoring")

        # Analyze code for refactoring opportunities
        structure = self.analyzer.parse_code_structure(
            request.existing_code, request.language
        )
        improvements = self.analyzer.suggest_improvements(
            structure, request.existing_code
        )

        # Apply refactoring based on improvements
        refactored_code = self._apply_refactoring(
            request.existing_code, improvements, request.language
        )

        return GeneratedCode(
            code=refactored_code,
            language=request.language,
            type=GenerationType.REFACTOR,
            confidence=0.7,
            description="Refactored code with improvements",
            alternatives=[request.existing_code],  # Original as alternative
            test_coverage_suggestions=[
                "Verify refactored code maintains same behavior",
                "Test performance improvements",
                "Ensure all edge cases still work",
            ],
        )

    async def _generate_bug_fix(self, request: GenerationRequest) -> GeneratedCode:
        """Generate bug fix for problematic code."""
        if not request.existing_code:
            raise ValueError("Existing code required for bug fix")

        # Analyze code for potential issues
        structure = self.analyzer.parse_code_structure(
            request.existing_code, request.language
        )
        patterns = self.analyzer.detect_code_patterns(structure, request.existing_code)

        # Generate fix based on detected issues
        fixed_code = self._generate_fixes(
            request.existing_code, patterns, request.language, request.description
        )

        return GeneratedCode(
            code=fixed_code,
            language=request.language,
            type=GenerationType.BUG_FIX,
            confidence=0.6,
            description=f"Bug fix: {request.description}",
            alternatives=[request.existing_code],
            test_coverage_suggestions=[
                "Add test case for the specific bug",
                "Test related functionality",
                "Add regression tests",
            ],
        )

    async def _optimize_code(self, request: GenerationRequest) -> GeneratedCode:
        """Optimize code for better performance."""
        if not request.existing_code:
            raise ValueError("Existing code required for optimization")

        # Focus on performance improvements
        optimized_code = self._apply_performance_optimizations(
            request.existing_code, request.language
        )

        return GeneratedCode(
            code=optimized_code,
            language=request.language,
            type=GenerationType.OPTIMIZATION,
            confidence=0.6,
            description="Performance-optimized code",
            alternatives=[request.existing_code],
            test_coverage_suggestions=[
                "Benchmark performance improvements",
                "Test that optimization doesn't break functionality",
                "Profile memory usage",
            ],
        )

    async def _generate_boilerplate(self, request: GenerationRequest) -> GeneratedCode:
        """Generate boilerplate code for common patterns."""
        boilerplate_type = request.parameters.get("boilerplate_type", "basic")

        if request.language == CodeLanguage.PYTHON:
            code = self._generate_python_boilerplate(boilerplate_type)
        elif request.language == CodeLanguage.JAVASCRIPT:
            code = self._generate_javascript_boilerplate(boilerplate_type)
        else:
            code = self._generate_generic_boilerplate(
                boilerplate_type, request.language
            )

        return GeneratedCode(
            code=code,
            language=request.language,
            type=GenerationType.BOILERPLATE,
            confidence=0.9,
            description=f"Generated {boilerplate_type} boilerplate",
            imports_needed=self._determine_imports(code, request.language),
        )

    def _parse_function_description(self, description: str) -> Dict[str, Any]:
        """Parse function description to extract name, params, etc."""
        # Simple regex-based parsing (could be enhanced with NLP)
        name_match = re.search(r"(?:function|def)\s+(\w+)", description)
        name = name_match.group(1) if name_match else "generated_function"

        # Extract parameters (very basic)
        param_pattern = r"(?:takes?|parameters?|args?)\s+([^.]+)"
        param_match = re.search(param_pattern, description, re.IGNORECASE)
        params = []
        if param_match:
            param_text = param_match.group(1)
            params = [p.strip() for p in re.split(r"[,and]", param_text) if p.strip()]

        # Extract return type
        return_pattern = r"(?:returns?|outputs?)\s+([^.]+)"
        return_match = re.search(return_pattern, description, re.IGNORECASE)
        return_type = return_match.group(1).strip() if return_match else "Any"

        return {
            "name": name,
            "params": params,
            "return_type": return_type,
            "description": description,
        }

    def _parse_class_description(self, description: str) -> Dict[str, Any]:
        """Parse class description to extract name, methods, etc."""
        name_match = re.search(r"(?:class)\s+(\w+)", description)
        name = name_match.group(1) if name_match else "GeneratedClass"

        # Extract methods (basic)
        method_pattern = r"(?:methods?|functions?)\s+([^.]+)"
        method_match = re.search(method_pattern, description, re.IGNORECASE)
        methods = []
        if method_match:
            method_text = method_match.group(1)
            methods = [m.strip() for m in re.split(r"[,and]", method_text) if m.strip()]

        return {"name": name, "methods": methods, "description": description}

    def _generate_function_body(
        self, func_info: Dict[str, Any], language: CodeLanguage, style: Dict[str, Any]
    ) -> str:
        """Generate function body based on description."""
        # This is a simplified implementation
        # In practice, this could use ML models or more sophisticated heuristics

        description = func_info.get("description", "")

        if "calculate" in description.lower() or "compute" in description.lower():
            if language == CodeLanguage.PYTHON:
                return "    # TODO: Implement calculation logic\n    pass"
            else:
                return "    // TODO: Implement calculation logic\n    return null;"
        elif "validate" in description.lower() or "check" in description.lower():
            if language == CodeLanguage.PYTHON:
                return "    # TODO: Implement validation logic\n    return True"
            else:
                return "    // TODO: Implement validation logic\n    return true;"
        else:
            if language == CodeLanguage.PYTHON:
                return "    # TODO: Implement function logic\n    pass"
            else:
                return "    // TODO: Implement function logic"

    def _generate_python_function(
        self, func_info: Dict[str, Any], body: str, style: Dict[str, Any]
    ) -> str:
        """Generate Python function code."""
        name = func_info.get("name", "generated_function")
        params = func_info.get("params", [])
        return_type = func_info.get("return_type", "Any")
        description = func_info.get("description", "Generated function")

        # Format parameters with type hints
        param_list = []
        for param in params:
            if style.get("use_type_hints", True):
                param_list.append(f"{param}: Any")
            else:
                param_list.append(param)

        params_str = ", ".join(param_list)

        # Generate docstring
        args_doc = ""
        for param in params:
            args_doc += f"        {param}: Parameter description\n"

        return f'''def {name}({params_str}) -> {return_type}:
    """
    {description}
    
    Args:
{args_doc}    
    Returns:
        {return_type}: Return value description
    """
{body}'''

    def _generate_javascript_function(
        self, func_info: Dict[str, Any], body: str, style: Dict[str, Any]
    ) -> str:
        """Generate JavaScript function code."""
        name = func_info.get("name", "generatedFunction")
        params = func_info.get("params", [])
        return_type = func_info.get("return_type", "any")
        description = func_info.get("description", "Generated function")

        params_str = ", ".join(params)

        # Generate JSDoc
        param_docs = ""
        for param in params:
            param_docs += f"     * @param {{any}} {param} Parameter description\n"

        return f"""function {name}({params_str}) {{
    /**
     * {description}
{param_docs}     * @returns {{{return_type}}} Return value description
     */
{body}
}}"""

    def _generate_generic_function(
        self, func_info: Dict[str, Any], body: str, language: CodeLanguage
    ) -> str:
        """Generate function for other languages."""
        name = func_info.get("name", "generated_function")
        params = ", ".join(func_info.get("params", []))

        return f"""// Generated function for {language.value}
// {func_info.get('description', 'Generated function')}
function {name}({params}) {{
{body}
}}"""

    def _generate_python_class(
        self, class_info: Dict[str, Any], style: Dict[str, Any]
    ) -> str:
        """Generate Python class code."""
        name = class_info.get("name", "GeneratedClass")
        description = class_info.get("description", "Generated class")
        methods = class_info.get("methods", [])

        init_body = "        pass"
        method_code = ""

        for method in methods:
            method_code += f"""
    def {method}(self):
        \"\"\"Method: {method}\"\"\"
        pass
"""

        return f'''class {name}:
    """
    {description}
    """
    
    def __init__(self):
        """Initialize {name}."""
{init_body}{method_code}'''

    def _generate_javascript_class(
        self, class_info: Dict[str, Any], style: Dict[str, Any]
    ) -> str:
        """Generate JavaScript class code."""
        name = class_info.get("name", "GeneratedClass")
        description = class_info.get("description", "Generated class")
        methods = class_info.get("methods", [])

        method_code = ""
        for method in methods:
            method_code += f"""
    {method}() {{
        // Method: {method}
    }}
"""

        return f"""class {name} {{
    /**
     * {description}
     */
    constructor() {{
        // Initialize class
    }}{method_code}
}}"""

    def _generate_generic_class(
        self, class_info: Dict[str, Any], language: CodeLanguage
    ) -> str:
        """Generate class for other languages."""
        name = class_info.get("name", "GeneratedClass")
        description = class_info.get("description", "Generated class")

        return f"""// Generated class for {language.value}
// {description}
class {name} {{
    // Constructor and methods would go here
}}"""

    def _get_style_preferences(
        self, language: CodeLanguage, custom_prefs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get style preferences for the language."""
        defaults = self.default_styles.get(language, {})
        return {**defaults, **custom_prefs}

    def _determine_imports(self, code: str, language: CodeLanguage) -> List[str]:
        """Determine what imports are needed for the generated code."""
        imports = []

        if language == CodeLanguage.PYTHON:
            if "typing" in code or "Any" in code:
                imports.append("from typing import Any")
            if "Optional" in code:
                imports.append("from typing import Optional")
            if "List" in code:
                imports.append("from typing import List")

        return imports

    def _determine_test_imports(self, language: CodeLanguage) -> List[str]:
        """Determine imports needed for test code."""
        if language == CodeLanguage.PYTHON:
            return ["import pytest"]
        elif language == CodeLanguage.JAVASCRIPT:
            return ["// No imports needed for basic tests"]
        else:
            return []

    def _estimate_complexity(self, code: str) -> int:
        """Estimate code complexity (very basic)."""
        # Count control flow statements
        complexity = 1  # Base complexity

        # Count if statements, loops, etc.
        complexity += len(
            re.findall(r"\b(if|for|while|elif|else|try|except|catch)\b", code)
        )

        return complexity

    def _generate_test_cases(
        self, structure: CodeStructure, language: CodeLanguage
    ) -> str:
        """Generate test cases for code structure."""
        if language == CodeLanguage.PYTHON:
            return self._generate_python_tests(structure)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._generate_javascript_tests(structure)
        else:
            return f"# Test cases for {language.value} (not implemented)"

    def _generate_python_tests(self, structure: CodeStructure) -> str:
        """Generate Python test cases."""
        test_code = "import pytest\n\n"

        for func in structure.functions:
            func_name = func["name"]
            test_code += f"""def test_{func_name}():
    \"\"\"Test {func_name} function.\"\"\"
    # TODO: Implement test for {func_name}
    pass

"""

        return test_code

    def _generate_javascript_tests(self, structure: CodeStructure) -> str:
        """Generate JavaScript test cases."""
        test_code = ""

        for func in structure.functions:
            func_name = func["name"]
            test_code += f"""describe('{func_name}', () => {{
    test('should work correctly', () => {{
        // TODO: Implement test for {func_name}
    }});
}});

"""

        return test_code

    def _generate_python_docstring(self, code: str) -> str:
        """Generate Python docstring for code."""
        # Simple docstring generation
        return '''"""
    Generated docstring.
    
    TODO: Add proper description, parameters, and return value documentation.
    """'''

    def _generate_jsdoc(self, code: str) -> str:
        """Generate JSDoc for JavaScript/TypeScript code."""
        return """/**
 * Generated JSDoc documentation.
 * 
 * TODO: Add proper description, parameters, and return value documentation.
 */"""

    def _generate_generic_documentation(self, code: str) -> str:
        """Generate documentation for other languages."""
        return """/*
 * Generated documentation.
 * 
 * TODO: Add proper description.
 */"""

    def _apply_refactoring(
        self, code: str, improvements: List[CodeImprovement], language: CodeLanguage
    ) -> str:
        """Apply refactoring based on improvement suggestions."""
        # This is a simplified implementation
        # In practice, this would apply specific refactoring transformations

        refactored = code

        for improvement in improvements:
            if improvement.improvement_type == ImprovementType.READABILITY:
                # Add comments for complex sections
                refactored = self._add_explanatory_comments(refactored, language)
            elif improvement.improvement_type == ImprovementType.PERFORMANCE:
                # Apply basic performance optimizations
                refactored = self._apply_basic_optimizations(refactored, language)

        return refactored

    def _add_explanatory_comments(self, code: str, language: CodeLanguage) -> str:
        """Add explanatory comments to complex code."""
        # Very basic implementation
        if language == CodeLanguage.PYTHON:
            return code.replace("def ", "# Function definition\n    def ")
        else:
            return code

    def _apply_basic_optimizations(self, code: str, language: CodeLanguage) -> str:
        """Apply basic performance optimizations."""
        # Very basic implementation
        return code

    def _generate_fixes(
        self, code: str, patterns: List[Any], language: CodeLanguage, description: str
    ) -> str:
        """Generate bug fixes based on detected patterns."""
        # Simplified implementation
        fixed_code = code

        # Add basic error handling if missing
        if (
            language == CodeLanguage.PYTHON
            and "try:" not in code
            and "except" not in code
        ):
            lines = code.split("\n")
            indented_lines = ["    " + line for line in lines if line.strip()]
            fixed_code = (
                "try:\n"
                + "\n".join(indented_lines)
                + "\nexcept Exception as e:\n    # Handle error\n    raise"
            )

        return fixed_code

    def _apply_performance_optimizations(
        self, code: str, language: CodeLanguage
    ) -> str:
        """Apply performance optimizations."""
        # Simplified implementation
        optimized = code

        if language == CodeLanguage.PYTHON:
            # Replace list concatenation in loops with list comprehension
            if "for " in code and ".append(" in code:
                optimized = (
                    code
                    + "\n# TODO: Consider using list comprehension for better performance"
                )

        return optimized

    def _generate_python_boilerplate(self, boilerplate_type: str) -> str:
        """Generate Python boilerplate code."""
        if boilerplate_type == "cli":
            return '''#!/usr/bin/env python3
"""
Command Line Interface module.
"""

import argparse
import sys
from typing import List, Optional


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CLI application")
    parser.add_argument("--version", action="version", version="1.0.0")
    
    parsed_args = parser.parse_args(args)
    
    # TODO: Implement CLI logic
    print("Hello, World!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())'''

        elif boilerplate_type == "class":
            return '''from typing import Any, Optional


class BaseClass:
    """Base class template."""
    
    def __init__(self, name: str, **kwargs: Any):
        """Initialize the class."""
        self.name = name
        self._config = kwargs
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()'''

        else:
            return '''"""
Generated Python module.
"""

# TODO: Add your code here

def main():
    """Main function."""
    pass


if __name__ == "__main__":
    main()'''

    def _generate_javascript_boilerplate(self, boilerplate_type: str) -> str:
        """Generate JavaScript boilerplate code."""
        if boilerplate_type == "node":
            return """#!/usr/bin/env node

/**
 * Node.js application template.
 */

const process = require('process');

function main(args = []) {
    // TODO: Implement application logic
    console.log('Hello, World!');
    return 0;
}

if (require.main === module) {
    const exitCode = main(process.argv.slice(2));
    process.exit(exitCode);
}

module.exports = { main };"""

        elif boilerplate_type == "class":
            return """/**
 * Base class template.
 */
class BaseClass {
    constructor(name, options = {}) {
        this.name = name;
        this._options = options;
    }
    
    toString() {
        return `${this.constructor.name}(name=${this.name})`;
    }
}

module.exports = BaseClass;"""

        else:
            return """/**
 * Generated JavaScript module.
 */

// TODO: Add your code here

function main() {
    console.log('Hello, World!');
}

if (require.main === module) {
    main();
}

module.exports = { main };"""

    def _generate_generic_boilerplate(
        self, boilerplate_type: str, language: CodeLanguage
    ) -> str:
        """Generate boilerplate for other languages."""
        return f"""// Generated boilerplate for {language.value}
// Type: {boilerplate_type}

// TODO: Add your code here

int main() {{
    // Main function
    return 0;
}}"""


# Factory function
def create_code_generator(
    analyzer: Optional[CodeAnalysisEngine] = None,
) -> CodeGenerator:
    """Create a code generator with default configuration."""
    return CodeGenerator(analyzer)


# Helper functions for easy usage
async def generate_function(
    description: str, language: CodeLanguage = CodeLanguage.PYTHON
) -> str:
    """Quick function generation."""
    generator = create_code_generator()
    request = GenerationRequest(
        type=GenerationType.FUNCTION, description=description, language=language
    )
    result = await generator.generate_code(request)
    return result.code


async def generate_tests_for_code(
    code: str, language: CodeLanguage = CodeLanguage.PYTHON
) -> str:
    """Quick test generation for existing code."""
    generator = create_code_generator()
    request = GenerationRequest(
        type=GenerationType.TEST,
        description="Generate unit tests",
        language=language,
        existing_code=code,
    )
    result = await generator.generate_code(request)
    return result.code


async def refactor_code(code: str, language: CodeLanguage = CodeLanguage.PYTHON) -> str:
    """Quick code refactoring."""
    generator = create_code_generator()
    request = GenerationRequest(
        type=GenerationType.REFACTOR,
        description="Refactor for better readability and performance",
        language=language,
        existing_code=code,
    )
    result = await generator.generate_code(request)
    return result.code
