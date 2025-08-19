#!/usr/bin/env python3
"""
OpenAgent Chatbot Example

This example demonstrates how to create a simple chatbot using OpenAgent
with different capabilities including file operations, system commands,
and conversation history.

Usage:
    python examples/chatbot_example.py

Features:
- Interactive conversation with AI
- File management capabilities
- System information retrieval
- Command execution (safe mode)
- Conversation history
- Graceful error handling
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openagent.core.agent import Agent
from openagent.core.config import Config
from openagent.tools.system import CommandExecutor, FileManager, SystemInfo
from openagent.tools.git import GitTool
from openagent.core.exceptions import AgentError, ModelLoadError
from openagent.core.history import HistoryManager


class ChatbotExample:
    """Simple chatbot example using OpenAgent."""
    
    def __init__(self, model_name: str = "tiny-llama", explain_only: bool = True):
        """
        Initialize the chatbot.
        
        Args:
            model_name: Name of the LLM model to use
            explain_only: If True, commands are explained but not executed
        """
        self.model_name = model_name
        self.explain_only = explain_only
        self.agent = None
        self.history_manager = HistoryManager()
        self.conversation_id = None
    
    async def initialize(self):
        """Initialize the agent and tools."""
        try:
            print("🚀 Initializing OpenAgent Chatbot...")
            
            # Create tools with safe configuration
            tools = [
                CommandExecutor(default_explain_only=self.explain_only),
                FileManager(safe_paths=["/tmp", str(Path.home() / "Documents")]),
                SystemInfo(),
                GitTool()
            ]
            
            # Configure the agent
            config = Config({
                "model": self.model_name,
                "device": "cpu",  # Use CPU for compatibility
                "load_in_4bit": False,
                "explain_only": self.explain_only,
                "log_level": "INFO"
            })
            
            # Create the agent
            self.agent = Agent(
                name="ChatbotAssistant",
                description="A helpful AI assistant with system capabilities",
                model_name=self.model_name,
                tools=tools,
                config=config
            )
            
            print(f"✅ Agent initialized with model: {self.model_name}")
            print(f"🔧 Tools available: {', '.join(tool.name for tool in tools)}")
            print(f"🔒 Safe mode: {'ON' if self.explain_only else 'OFF'}")
            
        except ModelLoadError as e:
            print(f"❌ Failed to load model '{self.model_name}': {e}")
            print("💡 Try using a different model or ensure the model is available")
            raise
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")
            raise
    
    async def start_conversation(self):
        """Start the interactive conversation loop."""
        if not self.agent:
            await self.initialize()
        
        print("\n" + "="*60)
        print("🤖 OpenAgent Chatbot Ready!")
        print("="*60)
        print("💬 Type your messages below")
        print("🔧 Available commands:")
        print("  /help     - Show this help message")
        print("  /tools    - List available tools")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /safe     - Toggle safe mode")
        print("  /exit     - Exit the chatbot")
        print("="*60)
        
        conversation_active = True
        
        while conversation_active:
            try:
                # Get user input
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    conversation_active = await self._handle_command(user_input)
                    continue
                
                # Process the message with the agent
                print("🤖 Assistant: ", end="", flush=True)
                
                response = await self.agent.process_message(user_input)
                print(response.content)
                
                # Save to history if available
                try:
                    history_block = HistoryManager.new_block(
                        input_text=user_input,
                        response=response.content,
                        model={"model_name": self.model_name},
                        metadata=response.metadata
                    )
                    self.history_manager.append(history_block)
                except Exception as e:
                    print(f"⚠️ Failed to save history: {e}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                conversation_active = False
            except AgentError as e:
                print(f"\n❌ Agent error: {e}")
                print("💡 Try rephrasing your request or check your input")
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("💡 The chatbot will continue running")
    
    async def _handle_command(self, command: str) -> bool:
        """
        Handle special chatbot commands.
        
        Args:
            command: The command string starting with '/'
            
        Returns:
            bool: True to continue conversation, False to exit
        """
        command = command.lower().strip()
        
        if command == '/help':
            print("\n📚 OpenAgent Chatbot Help:")
            print("  - Ask questions about anything")
            print("  - Request file operations (reading, listing)")
            print("  - Get system information")
            print("  - Execute commands (in safe mode by default)")
            print("  - Get help with git operations")
            print("  - Use natural language for all requests")
            print("\n💡 Examples:")
            print("  'List files in my home directory'")
            print("  'What's my system's CPU information?'")
            print("  'Create a Python script that prints hello world'")
            print("  'Show me the git status of this repository'")
            
        elif command == '/tools':
            print("\n🔧 Available Tools:")
            if self.agent:
                for tool in self.agent.tools.values():
                    print(f"  • {tool.name}: {tool.description}")
            else:
                print("  Agent not initialized")
                
        elif command == '/history':
            print("\n📚 Conversation History:")
            try:
                blocks = self.history_manager.list_blocks(limit=5)
                if blocks:
                    for i, block in enumerate(blocks, 1):
                        print(f"\n{i}. 🧑 {block['input']}")
                        print(f"   🤖 {block['response'][:100]}{'...' if len(block['response']) > 100 else ''}")
                else:
                    print("  No conversation history available")
            except Exception as e:
                print(f"  ❌ Error retrieving history: {e}")
                
        elif command == '/clear':
            try:
                # Clear in-memory conversation state
                if self.agent:
                    self.agent.clear_context()
                print("✅ Conversation history cleared")
            except Exception as e:
                print(f"❌ Error clearing history: {e}")
                
        elif command == '/safe':
            if self.agent:
                current_mode = self.explain_only
                self.explain_only = not current_mode
                # Update command executor
                cmd_tool = self.agent.get_tool("command_executor")
                if cmd_tool:
                    cmd_tool.default_explain_only = self.explain_only
                print(f"🔒 Safe mode: {'ON' if self.explain_only else 'OFF'}")
            else:
                print("❌ Agent not initialized")
                
        elif command == '/exit':
            print("👋 Thank you for using OpenAgent Chatbot!")
            return False
            
        else:
            print(f"❌ Unknown command: {command}")
            print("💡 Type '/help' for available commands")
        
        return True


async def main():
    """Main function to run the chatbot example."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenAgent Chatbot Example")
    parser.add_argument(
        "--model", 
        default="tiny-llama",
        help="Model name to use (default: tiny-llama)"
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Disable safe mode (commands will be executed)"
    )
    parser.add_argument(
        "--example-queries",
        action="store_true",
        help="Show example queries and exit"
    )
    
    args = parser.parse_args()
    
    if args.example_queries:
        print("📝 Example Queries for OpenAgent Chatbot:")
        print("\n🔍 System Information:")
        print("  • What's my system's CPU and memory information?")
        print("  • Show me the current system load")
        print("  • What operating system am I running?")
        
        print("\n📁 File Operations:")
        print("  • List files in my Documents folder")
        print("  • Create a new text file with a hello world message")
        print("  • Read the contents of package.json if it exists")
        
        print("\n💻 Command Execution:")
        print("  • Show me the current directory")
        print("  • List all Python files in this directory")
        print("  • Check if Docker is installed")
        
        print("\n🔄 Git Operations:")
        print("  • What's the git status of this repository?")
        print("  • Show me the latest git commits")
        print("  • Check which branch I'm currently on")
        
        print("\n🤖 General AI Assistance:")
        print("  • Explain how to set up a Python virtual environment")
        print("  • What are the best practices for code documentation?")
        print("  • Help me write a Python function to calculate fibonacci numbers")
        
        return
    
    # Create and run the chatbot
    chatbot = ChatbotExample(
        model_name=args.model,
        explain_only=not args.unsafe
    )
    
    try:
        await chatbot.start_conversation()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error running chatbot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the chatbot
    asyncio.run(main())
