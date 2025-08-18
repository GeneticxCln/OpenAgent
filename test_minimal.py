#!/usr/bin/env python3
"""
Minimal test to verify basic openagent functionality without heavy dependencies
"""

def test_basic_imports():
    """Test that basic imports work"""
    try:
        import openagent
        print("✅ openagent import successful")
        
        from openagent.core.agent import Agent
        from openagent.core.base import BaseMessage
        print("✅ Core imports successful")
        
        # Test agent creation
        agent = Agent("test_agent")
        print(f"✅ Agent created: {agent.name}")
        
        # Test basic message processing (should use fallback)
        import asyncio
        
        async def test_agent():
            message = await agent.process_message("Hello")
            print(f"✅ Agent response: {message.content[:50]}...")
            return True
            
        result = asyncio.run(test_agent())
        assert result, "Agent processing failed"
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_imports()
    exit(0 if success else 1)
