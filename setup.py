#!/usr/bin/env python3
"""
Setup script for OpenAgent development environment.

This script helps set up the development environment and install dependencies.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {command}")
    return subprocess.run(command, shell=True, check=check)


def setup_development():
    """Set up development environment."""
    print("🚀 Setting up OpenAgent development environment...")
    
    # Activate virtual environment and install in development mode
    print("\\n📦 Installing OpenAgent in development mode...")
    run_command("source venv/bin/activate && pip install -e .")
    
    print("\\n🔧 Installing development dependencies...")
    run_command("source venv/bin/activate && pip install -e \".[dev]\"")
    
    print("\\n📁 Creating necessary directories...")
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    
    print("\\n📄 Creating configuration file...")
    if not Path(".env").exists():
        run_command("cp .env.example .env")
        print("Created .env file. Please edit it with your configuration.")
    
    print("\\n✅ Development environment setup complete!")
    print("\\nNext steps:")
    print("1. Edit .env file with your Hugging Face token (optional)")
    print("2. Run: source venv/bin/activate")
    print("3. Run: openagent chat")


def quick_test():
    """Run a quick test to verify everything works."""
    print("\\n🧪 Running quick test...")
    
    try:
        run_command("source venv/bin/activate && python -c \"from openagent import Agent; print('✅ Import successful')\"")
        run_command("source venv/bin/activate && openagent models")
        print("\\n✅ Quick test passed!")
    except subprocess.CalledProcessError:
        print("\\n❌ Quick test failed. Please check your installation.")
        return False
    
    return True


def install_pytorch():
    """Install PyTorch with CUDA support if available."""
    print("\\n🔥 Installing PyTorch...")
    
    try:
        # Try to detect CUDA
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            print("CUDA detected, installing PyTorch with CUDA support...")
            run_command("source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("No CUDA detected, installing CPU-only PyTorch...")
            run_command("source venv/bin/activate && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    except Exception as e:
        print(f"Error installing PyTorch: {e}")
        print("You may need to install PyTorch manually.")


def main():
    """Main setup function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "pytorch":
            install_pytorch()
        elif sys.argv[1] == "dev":
            setup_development()
        else:
            print("Usage: python setup.py [dev|test|pytorch]")
    else:
        setup_development()


if __name__ == "__main__":
    main()
