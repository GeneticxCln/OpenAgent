#!/usr/bin/env python3
"""
Simple demo of OpenAgent functionality without full installation.
This shows you what OpenAgent can do!
"""

def demo_command_explanation():
    """Demo of command explanation feature."""
    print("🔍 COMMAND EXPLANATION DEMO")
    print("=" * 50)
    
    commands = [
        "ls -la",
        "find . -name '*.py'",
        "docker build -t myapp .",
        "git commit -m 'Initial commit'",
        "ps aux | grep python"
    ]
    
    explanations = {
        "ls -la": """
Lists directory contents with detailed information:
- -l: Long format (permissions, size, date, etc.)
- -a: Show hidden files (starting with .)
Result: Shows all files with full details including hidden files
""",
        
        "find . -name '*.py'": """
Searches for files recursively:
- find: Search command
- . : Start from current directory
- -name '*.py': Find files matching pattern *.py
Result: Lists all Python files in current directory and subdirectories
""",
        
        "docker build -t myapp .": """
Builds a Docker image:
- docker build: Build image from Dockerfile
- -t myapp: Tag the image with name 'myapp'
- . : Use current directory as build context
Result: Creates Docker image named 'myapp' from current directory
""",
        
        "git commit -m 'Initial commit'": """
Commits changes to Git repository:
- git commit: Save staged changes
- -m: Add commit message inline
- 'Initial commit': The commit message
Result: Creates a new commit with the specified message
""",
        
        "ps aux | grep python": """
Shows running Python processes:
- ps aux: List all running processes
- |: Pipe output to next command  
- grep python: Filter for lines containing 'python'
Result: Shows all currently running Python processes
"""
    }
    
    for cmd in commands:
        print(f"💻 Command: {cmd}")
        print(f"📝 Explanation:{explanations[cmd]}")
        print("-" * 50)


def demo_code_generation():
    """Demo of code generation feature."""
    print("\n💻 CODE GENERATION DEMO")
    print("=" * 50)
    
    examples = {
        "Create a hello world function": '''
def hello_world(name="World"):
    """
    A simple hello world function.
    
    Args:
        name (str): Name to greet
        
    Returns:
        str: Greeting message
    """
    return f"Hello, {name}!"

# Example usage
if __name__ == "__main__":
    print(hello_world())
    print(hello_world("Alice"))
''',
        
        "Create a function to read CSV files": '''
import csv
from typing import List, Dict, Any

def read_csv_file(filename: str) -> List[Dict[str, Any]]:
    """
    Read a CSV file and return data as a list of dictionaries.
    
    Args:
        filename (str): Path to CSV file
        
    Returns:
        List[Dict[str, Any]]: List of rows as dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid CSV
    """
    try:
        data = []
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(dict(row))
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{filename}' not found")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

# Example usage
if __name__ == "__main__":
    try:
        data = read_csv_file("example.csv")
        print(f"Loaded {len(data)} rows")
        for row in data[:3]:  # Show first 3 rows
            print(row)
    except Exception as e:
        print(f"Error: {e}")
''',
        
        "Create a REST API with FastAPI": '''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="My API", version="1.0.0")

# Data models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float

class ItemResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    price: float

# In-memory storage (use database in production)
items_db: List[ItemResponse] = []
next_id = 1

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "API is running!"}

@app.get("/items/", response_model=List[ItemResponse])
async def get_items():
    """Get all items."""
    return items_db

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get item by ID."""
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/items/", response_model=ItemResponse)
async def create_item(item: Item):
    """Create a new item."""
    global next_id
    new_item = ItemResponse(
        id=next_id,
        name=item.name,
        description=item.description,
        price=item.price
    )
    items_db.append(new_item)
    next_id += 1
    return new_item

@app.put("/items/{item_id}", response_model=ItemResponse)
async def update_item(item_id: int, item: Item):
    """Update an existing item."""
    for i, existing_item in enumerate(items_db):
        if existing_item.id == item_id:
            updated_item = ItemResponse(
                id=item_id,
                name=item.name,
                description=item.description,
                price=item.price
            )
            items_db[i] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item."""
    for i, item in enumerate(items_db):
        if item.id == item_id:
            del items_db[i]
            return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    # Run with: python filename.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    }
    
    for description, code in examples.items():
        print(f"🎯 Request: {description}")
        print("📄 Generated Code:")
        print(f"```python{code}```")
        print("-" * 50)


def demo_system_info():
    """Demo of system information features."""
    print("\n🖥️ SYSTEM INFORMATION DEMO")
    print("=" * 50)
    
    import psutil
    import platform
    
    # System overview
    print("🔧 System Overview:")
    uname = platform.uname()
    print(f"  OS: {uname.system} {uname.release}")
    print(f"  Architecture: {uname.machine}")
    print(f"  Hostname: {uname.node}")
    
    # CPU info
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1)}%")
    print(f"  CPU Cores: {psutil.cpu_count()} physical, {psutil.cpu_count(logical=True)} logical")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"  Memory: {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent}%)")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"  Disk: {disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB ({disk.used / disk.total * 100:.1f}%)")
    
    print("\n📊 Top Processes:")
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Sort by CPU usage and show top 5
    processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)
    for proc in processes[:5]:
        print(f"  {proc['pid']:>6} {proc['name']:<20} CPU: {proc['cpu_percent']:>5.1f}% Memory: {proc['memory_percent']:>5.1f}%")


def demo_terminal_assistance():
    """Demo of terminal assistance features."""
    print("\n🚀 TERMINAL ASSISTANCE DEMO")
    print("=" * 50)
    
    scenarios = [
        {
            "user_request": "How do I find large files in my system?",
            "ai_response": """To find large files on your system, you can use several approaches:

1. **Find files larger than 100MB:**
   ```bash
   find / -type f -size +100M 2>/dev/null
   ```

2. **Find largest files in current directory:**
   ```bash
   find . -type f -exec ls -lh {} + | sort -k5 -hr | head -10
   ```

3. **Use du to find large directories:**
   ```bash
   du -h --max-depth=1 / 2>/dev/null | sort -hr | head -10
   ```

4. **Interactive file explorer (if available):**
   ```bash
   ncdu /  # Interactive disk usage analyzer
   ```

The first command is usually most useful for system cleanup!"""
        },
        
        {
            "user_request": "Help me set up a Python virtual environment",
            "ai_response": """Here's how to set up a Python virtual environment:

1. **Create a virtual environment:**
   ```bash
   python -m venv myenv
   # or for specific Python version:
   python3.11 -m venv myenv
   ```

2. **Activate the environment:**
   ```bash
   # On Linux/Mac:
   source myenv/bin/activate
   
   # On Windows:
   myenv\\Scripts\\activate
   ```

3. **Install packages:**
   ```bash
   pip install package_name
   pip install -r requirements.txt
   ```

4. **Deactivate when done:**
   ```bash
   deactivate
   ```

5. **Create requirements.txt:**
   ```bash
   pip freeze > requirements.txt
   ```

Your prompt will change to show (myenv) when the environment is active!"""
        },
        
        {
            "user_request": "How do I check if a service is running?",
            "ai_response": """Here are several ways to check if a service is running:

1. **Using systemctl (systemd systems):**
   ```bash
   systemctl status service_name
   systemctl is-active service_name
   systemctl is-enabled service_name
   ```

2. **Using ps to find process:**
   ```bash
   ps aux | grep service_name
   pgrep service_name
   ```

3. **Check specific ports:**
   ```bash
   netstat -tlnp | grep :port_number
   ss -tlnp | grep :port_number
   lsof -i :port_number
   ```

4. **Using service command:**
   ```bash
   service service_name status
   ```

5. **Check all running services:**
   ```bash
   systemctl list-units --type=service --state=running
   ```

For web services, you can also test with curl or wget!"""
        }
    ]
    
    for scenario in scenarios:
        print(f"❓ User: {scenario['user_request']}")
        print(f"🤖 Assistant: {scenario['ai_response']}")
        print("-" * 50)


def main():
    """Run all demos."""
    print("🎉 OPENAGENT FUNCTIONALITY DEMO")
    print("This shows what OpenAgent can do for you!")
    print("=" * 60)
    
    demo_command_explanation()
    demo_code_generation()
    demo_system_info()
    demo_terminal_assistance()
    
    print("\n✨ REAL USAGE:")
    print("Once properly installed, you would use OpenAgent like this:")
    print("")
    print("# Interactive chat")
    print("openagent chat --model tiny-llama")
    print("")
    print("# Quick code generation")
    print("openagent code 'Create a web scraper' --language python")
    print("")
    print("# Command explanation")
    print("openagent explain 'docker-compose up -d'")
    print("")
    print("# System assistance")
    print("openagent run 'How do I free up disk space?'")
    print("")
    print("🚀 This is a powerful AI assistant for developers and system administrators!")


if __name__ == "__main__":
    main()
