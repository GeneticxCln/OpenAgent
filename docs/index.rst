OpenAgent Documentation
=======================

.. image:: https://img.shields.io/pypi/v/openagent
   :target: https://pypi.org/project/openagent/
   :alt: PyPI version

.. image:: https://img.shields.io/github/stars/GeneticxCln/OpenAgent
   :target: https://github.com/GeneticxCln/OpenAgent
   :alt: GitHub stars

.. image:: https://img.shields.io/github/license/GeneticxCln/OpenAgent
   :target: https://github.com/GeneticxCln/OpenAgent/blob/main/LICENSE
   :alt: License

.. image:: https://codecov.io/gh/GeneticxCln/OpenAgent/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/GeneticxCln/OpenAgent
   :alt: Code coverage

**OpenAgent** is a powerful, production-ready AI agent framework powered by Hugging Face models, designed for terminal assistance, code generation, and system operations - your open source alternative to Warp AI.

🚀 **Key Features**
-------------------

* 🤖 **20+ AI Models**: CodeLlama, Mistral, Llama2, TinyLlama, and more
* 💻 **Terminal Assistant**: Execute commands safely, explain operations, manage files  
* 🔧 **Code Generation**: Generate, analyze, and review code in multiple languages
* ⚡ **High Performance**: 4-bit quantization, CUDA support, CPU fallback
* 🛠️ **Advanced Tools**: System monitoring, file management, command execution
* 🎨 **Rich CLI Interface**: Beautiful terminal UI with syntax highlighting
* 🔒 **Security First**: Safe command execution with built-in validation
* 📱 **Multiple Interfaces**: CLI chat, single commands, shell integration

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install from PyPI
   pip install openagent

   # Or install from source
   git clone https://github.com/GeneticxCln/OpenAgent.git
   cd OpenAgent
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Interactive chat with lightweight model
   openagent chat --model tiny-llama

   # Generate code
   openagent code "Create a hello world function" --language python

   # Explain commands
   openagent explain "docker ps -a"

   # System information
   openagent run "Show me system resources"

Python API
~~~~~~~~~~

.. code-block:: python

   from openagent import Agent
   from openagent.tools.system import CommandExecutor, SystemInfo
   
   # Create agent with tools
   agent = Agent(
       name="MyAssistant",
       model_name="codellama-7b",
       tools=[CommandExecutor(), SystemInfo()]
   )
   
   # Process messages
   response = await agent.process_message("List Python files in current directory")
   print(response.content)

🏗️ **Architecture**
-------------------

OpenAgent follows a modular architecture with clear separation of concerns:

.. mermaid::

   graph TD
       A[CLI Interface] --> B[Agent Core]
       B --> C[LLM Integration]
       B --> D[Tool System]
       B --> E[Configuration]
       
       C --> F[Hugging Face Models]
       D --> G[System Tools]
       D --> H[File Manager]
       D --> I[Command Executor]
       
       J[Terminal Integration] --> B
       K[Web API] --> B

Core Components:

* **Agent Core**: Main orchestration and message processing
* **LLM Integration**: Hugging Face model loading and inference
* **Tool System**: Extensible tools for system operations
* **CLI Interface**: Rich command-line interaction
* **Terminal Integration**: Native shell integration (zsh/bash)

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   models
   cli_reference
   shell_integration

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   api_reference
   plugin_development
   contributing
   testing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/tools
   api/llm
   api/cli

.. toctree::
   :maxdepth: 1
   :caption: Examples & Tutorials

   examples/basic_usage
   examples/custom_agents
   examples/tool_development
   examples/advanced_features

.. toctree::
   :maxdepth: 1
   :caption: About

   changelog
   roadmap
   license
   support

📊 **Performance & Compatibility**
----------------------------------

OpenAgent is optimized for performance and compatibility:

.. list-table:: System Requirements
   :header-rows: 1

   * - Component
     - Minimum
     - Recommended
   * - Python
     - 3.9+
     - 3.11+
   * - RAM
     - 4GB
     - 8GB+
   * - Storage
     - 2GB
     - 5GB+
   * - GPU
     - None (CPU works)
     - CUDA-compatible

Supported Platforms:

* ✅ Linux (Ubuntu, CentOS, Arch, etc.)
* ✅ macOS (Intel & Apple Silicon)
* ✅ Windows (WSL recommended)

🤝 **Community & Support**
--------------------------

Join our growing community:

* 💬 **GitHub Discussions**: `Ask questions and share ideas <https://github.com/GeneticxCln/OpenAgent/discussions>`_
* 🐛 **Issue Tracker**: `Report bugs and request features <https://github.com/GeneticxCln/OpenAgent/issues>`_
* 📧 **Email**: your.email@example.com
* 🐦 **Twitter**: @OpenAgentAI

Contributing:

* 🔧 **Development**: See our `contributing guide <https://github.com/GeneticxCln/OpenAgent/blob/main/CONTRIBUTING.md>`_
* 📝 **Documentation**: Help improve these docs
* 🧪 **Testing**: Add tests and report issues
* 💡 **Ideas**: Suggest features and improvements

📈 **Project Status**
---------------------

.. note::
   OpenAgent is in active development. We follow semantic versioning and maintain backward compatibility within major versions.

Current Status:

* **Latest Version**: v0.1.0
* **Development Status**: Alpha (Stable API)
* **Test Coverage**: 70%+
* **Platform Support**: Linux, macOS, Windows
* **Documentation**: Comprehensive

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
