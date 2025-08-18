# 🤝 Contributing to OpenAgent

Thank you for your interest in contributing to OpenAgent! This document provides guidelines and information to help you contribute effectively.

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Getting Started](#-getting-started)
- [Development Setup](#-development-setup)
- [Contributing Guidelines](#-contributing-guidelines)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Submitting Changes](#-submitting-changes)
- [Release Process](#-release-process)

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [your.email@example.com](mailto:your.email@example.com).

### Our Pledge

- **Be inclusive**: Welcome contributors from all backgrounds
- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together constructively
- **Be patient**: Help newcomers learn and grow
- **Focus on what's best for the community**

## 🚀 Getting Started

### Ways to Contribute

- 🐛 **Report bugs** - Help us identify and fix issues
- 💡 **Suggest features** - Propose new functionality
- 📝 **Improve documentation** - Help make docs clearer
- 🧪 **Write tests** - Improve test coverage
- 🔧 **Fix issues** - Submit pull requests
- 🎨 **Improve UX** - Enhance user experience
- 📦 **Package management** - Help with packaging and distribution

### First-Time Contributors

New to open source? No problem! Look for issues labeled `good-first-issue` or `help-wanted`. These are specifically chosen to be beginner-friendly.

## 💻 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/OpenAgent.git
   cd OpenAgent
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify installation**
   ```bash
   # Run tests
   pytest
   
   # Check code quality
   black --check .
   isort --check-only .
   flake8
   mypy openagent
   ```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest tests/
   black .
   isort .
   flake8
   mypy openagent
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

## 📋 Contributing Guidelines

### Coding Standards

#### Python Style
- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Maximum line length: 88 characters (Black default)

#### Code Quality
- **Type hints**: All functions must have type hints
- **Docstrings**: All public functions/classes need docstrings
- **Error handling**: Use appropriate exception handling
- **Logging**: Use structured logging with appropriate levels

#### Example Code Style
```python
"""
Module docstring describing the purpose.
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExampleClass:
    """Class docstring describing purpose and usage."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the example class.
        
        Args:
            name: The name of the instance
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        
    async def process_data(self, data: str) -> Optional[str]:
        """
        Process input data and return result.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data or None if processing fails
            
        Raises:
            ValueError: If data is invalid
        """
        if not data:
            raise ValueError("Data cannot be empty")
            
        try:
            result = self._internal_process(data)
            logger.info(f"Successfully processed data for {self.name}")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return None
            
    def _internal_process(self, data: str) -> str:
        """Internal processing method."""
        return data.upper()
```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `test`: Adding missing tests
- `chore`: Changes to build process or auxiliary tools

**Examples:**
- `feat(cli): add new chat command with streaming support`
- `fix(llm): handle CUDA out of memory errors gracefully`
- `docs: update installation instructions`
- `test: add integration tests for system tools`

## 🧪 Testing

### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Test file naming**: `test_*.py` or `*_test.py`
2. **Test function naming**: `test_descriptive_name`
3. **Use fixtures**: Leverage pytest fixtures for reusable test data
4. **Mock external dependencies**: Use `unittest.mock` for external services
5. **Async tests**: Use `@pytest.mark.asyncio` for async functions

#### Example Test
```python
import pytest
from unittest.mock import Mock, patch
from openagent.core.agent import Agent


class TestAgent:
    """Test suite for Agent class."""
    
    @pytest.fixture
    def mock_llm(self):
        """Fixture providing a mock LLM."""
        return Mock()
    
    def test_agent_initialization(self):
        """Test agent initialization with valid parameters."""
        agent = Agent(name="TestAgent", model_name="tiny-llama")
        
        assert agent.name == "TestAgent"
        assert agent.model_name == "tiny-llama"
        assert len(agent.tools) == 0
    
    @pytest.mark.asyncio
    async def test_process_message(self, mock_llm):
        """Test message processing functionality."""
        with patch('openagent.core.agent.HuggingFaceLLM', return_value=mock_llm):
            mock_llm.generate_response.return_value = "Test response"
            
            agent = Agent(name="TestAgent", model_name="tiny-llama")
            response = await agent.process_message("Hello")
            
            assert response.content == "Test response"
            mock_llm.generate_response.assert_called_once()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_core_agent.py

# Run with coverage
pytest --cov=openagent

# Run integration tests only
pytest -m integration

# Run tests in parallel
pytest -n auto
```

### Coverage Requirements

- **Minimum coverage**: 70%
- **New code coverage**: 90%+
- **Critical paths**: 100% coverage required

## 📚 Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Developer Documentation**: Technical implementation details
4. **README**: Project overview and quick start

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add screenshots for UI features
- Keep documentation up-to-date with code changes

### Documentation Tools

- **Sphinx**: API documentation generation
- **MkDocs**: User-friendly documentation site
- **Docstrings**: In-code documentation

## 📤 Submitting Changes

### Pull Request Process

1. **Ensure tests pass**
   ```bash
   pytest
   black --check .
   isort --check-only .
   flake8
   mypy openagent
   ```

2. **Update documentation** if needed

3. **Create pull request**
   - Use descriptive title
   - Reference related issues
   - Include testing notes
   - Add screenshots for UI changes

4. **Pull request template**
   ```markdown
   ## 📋 Description
   Brief description of changes
   
   ## 🔗 Related Issues
   Fixes #123
   
   ## 🧪 Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## 📝 Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added for new functionality
   ```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in CI/CD pipeline
4. **Approval** from at least one maintainer
5. **Merge** using squash and merge

## 🏷️ Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/) (SemVer):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. **Version bump** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release branch**
4. **Tag release** with version number
5. **Deploy** through CI/CD pipeline

## 🏆 Recognition

Contributors are recognized in:
- **README.md** - Hall of Fame section
- **Release notes** - Contributor acknowledgments
- **GitHub contributors** page
- **Annual contributor report**

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord/Slack**: Real-time chat (if available)
- **Email**: [your.email@example.com](mailto:your.email@example.com)

### Support

- **Documentation**: Check existing docs first
- **Search Issues**: Look for existing solutions
- **Ask Questions**: Use GitHub Discussions
- **Report Bugs**: Use GitHub Issues with template

## 🙏 Thank You

Your contributions make OpenAgent better for everyone! Whether you're fixing a typo or implementing a major feature, every contribution is valued and appreciated.

---

**Happy Contributing! 🎉**
