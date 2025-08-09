OpenAgent Documentation
=======================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/github/license/GeneticxCln/OpenAgent
   :target: https://github.com/GeneticxCln/OpenAgent/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/stars/GeneticxCln/OpenAgent
   :target: https://github.com/GeneticxCln/OpenAgent
   :alt: GitHub Stars

**OpenAgent** is a powerful, production-ready AI agent framework powered by Hugging Face models, designed for terminal assistance, code generation, and system operations - just like Warp AI but open source and customizable.

🚀 Features
-----------

* **🤖 Hugging Face Integration** - Use any HF model (CodeLlama, Mistral, Llama2, etc.)
* **💻 Terminal Assistant** - Execute commands safely, explain operations, manage files
* **🔧 Code Generation** - Generate, analyze, and review code in multiple languages
* **⚡ High Performance** - Optimized with 4-bit quantization, CUDA support
* **🛠️ Advanced Tools** - System monitoring, file management, command execution
* **🎨 Rich CLI Interface** - Beautiful terminal interface with syntax highlighting
* **🔒 Security First** - Safe command execution with built-in security checks
* **📱 Multiple Interfaces** - CLI chat and single commands (API server planned)

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/GeneticxCln/OpenAgent.git
   cd OpenAgent
   source venv/bin/activate
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Interactive chat
   openagent chat --model tiny-llama

   # Single command
   openagent run "How do I list Python files?"

   # Generate code
   openagent code "Create a function to calculate fibonacci" --language python

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   configuration
   models

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/tools
   api/cli
   api/server

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   development/contributing
   development/plugins
   development/architecture

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/custom_tools
   examples/advanced_workflows

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
