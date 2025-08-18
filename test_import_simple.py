#!/usr/bin/env python3
"""
Very simple import test to isolate the problem
"""

print("Step 1: Testing basic imports...")

try:
    import sys
    print(f"✅ Python version: {sys.version}")
except Exception as e:
    print(f"❌ sys import failed: {e}")

try:
    from openagent.core.exceptions import OpenAgentError
    print("✅ exceptions imported successfully")
except Exception as e:
    print(f"❌ exceptions import failed: {e}")

try:
    from openagent.core.base import BaseAgent
    print("✅ base imported successfully")
except Exception as e:
    print(f"❌ base import failed: {e}")

try:
    from openagent.core.config import Config
    print("✅ config imported successfully")
except Exception as e:
    print(f"❌ config import failed: {e}")

try:
    from openagent.core.agent import Agent
    print("✅ agent imported successfully")
except Exception as e:
    print(f"❌ agent import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from openagent import Agent as TopLevelAgent
    print("✅ top-level agent imported successfully")
except Exception as e:
    print(f"❌ top-level agent import failed: {e}")
    import traceback
    traceback.print_exc()

print("Import test complete.")
