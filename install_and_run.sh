#!/bin/bash

echo "🚀 Installing OpenAgent..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Install PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install required dependencies
echo "📚 Installing dependencies..."
pip install transformers huggingface-hub tokenizers accelerate
pip install pydantic fastapi rich typer psutil

# Install OpenAgent
echo "🤖 Installing OpenAgent..."
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎯 How to use OpenAgent:"
echo ""
echo "1. Interactive chat:"
echo "   openagent chat --model tiny-llama"
echo ""
echo "2. Generate code:"
echo "   openagent code 'Create a hello world function' --language python"
echo ""
echo "3. Explain commands:"
echo "   openagent explain 'ls -la'"
echo ""
echo "4. Quick assistance:"
echo "   openagent run 'How do I check disk usage?'"
echo ""
echo "5. List available models:"
echo "   openagent models"
echo ""
echo "🚀 Try it now: openagent chat --model tiny-llama"
