#!/usr/bin/env python3
"""
Non-packaging helper script placeholder.

This repository uses pyproject.toml with setuptools.build_meta for packaging.
This file intentionally does nothing to avoid interfering with PEP 517 builds.

Developer helpers previously lived here and have been moved to scripts/ if needed.
"""

# Intentionally a no-op to prevent setuptools from invoking custom logic during builds.
# Keep this file minimal so editable installs (`pip install -e .`) work reliably.

if __name__ == "__main__":
    print("OpenAgent: setup.py is a no-op. Use: pip install -e . or see pyproject.toml.")
