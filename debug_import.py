#!/usr/bin/env python3
"""
Debug script to test which imports are failing in CI
"""

print("🔍 Testing imports...")

try:
    import openagent
    print("✅ openagent imported successfully")
except ImportError as e:
    print(f"❌ openagent import failed: {e}")

try:
    from openagent.core.agent import Agent
    print("✅ Agent imported successfully")
except ImportError as e:
    print(f"❌ Agent import failed: {e}")

try:
    from openagent.core.history import HistoryManager
    print("✅ HistoryManager imported successfully")
except ImportError as e:
    print(f"❌ HistoryManager import failed: {e}")

try:
    from openagent.core.tool_selector import SmartToolSelector
    print("✅ SmartToolSelector imported successfully")
except ImportError as e:
    print(f"❌ SmartToolSelector import failed: {e}")

try:
    from openagent.tools.system import CommandExecutor
    print("✅ CommandExecutor imported successfully")
except ImportError as e:
    print(f"❌ CommandExecutor import failed: {e}")

try:
    import pytest
    print("✅ pytest imported successfully")
except ImportError as e:
    print(f"❌ pytest import failed: {e}")

print("🔍 Import test complete")
