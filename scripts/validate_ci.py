#!/usr/bin/env python3
"""
Validate CI/CD workflows and project configuration.
"""

import os
import sys
import json
from pathlib import Path

def validate_yaml_syntax(file_path):
    """Validate YAML file syntax using basic checks."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Basic YAML syntax checks
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Check for basic indentation consistency
                if line.startswith(' ') and not line.startswith('  '):
                    if len(line) - len(line.lstrip()) % 2 != 0:
                        print(f"Warning: Inconsistent indentation at line {i} in {file_path}")
                
        print(f"✅ {file_path} - Basic YAML validation passed")
        return True
    except Exception as e:
        print(f"❌ {file_path} - YAML validation failed: {e}")
        return False

def check_file_exists(file_path, description=""):
    """Check if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description or file_path} exists")
        return True
    else:
        print(f"❌ {description or file_path} is missing")
        return False

def validate_scripts():
    """Validate that required scripts exist and are executable."""
    scripts_dir = Path("scripts")
    required_scripts = [
        "build_binary.sh",
        "export_openapi.sh",
    ]
    
    all_good = True
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            if os.access(script_path, os.X_OK):
                print(f"✅ {script} exists and is executable")
            else:
                print(f"⚠️ {script} exists but is not executable")
                all_good = False
        else:
            print(f"❌ {script} is missing")
            all_good = False
    
    return all_good

def main():
    """Main validation function."""
    print("🔍 Validating OpenAgent CI/CD Pipeline...\n")
    
    # Change to project root
    os.chdir(Path(__file__).parent.parent)
    
    all_checks_passed = True
    
    # Validate workflow files
    workflows = [
        ".github/workflows/ci.yml",
        ".github/workflows/docs.yml",
        ".github/workflows/fast-ci.yml",
        ".github/workflows/nightly.yml",
    ]
    
    print("📋 Validating workflow files:")
    for workflow in workflows:
        if not validate_yaml_syntax(workflow):
            all_checks_passed = False
    print()
    
    # Check required configuration files
    print("📁 Checking configuration files:")
    config_files = [
        ("pyproject.toml", "Project configuration"),
        ("mkdocs.yml", "Documentation configuration"), 
        ("requirements-dev.txt", "Development dependencies"),
        (".pre-commit-config.yaml", "Pre-commit configuration"),
    ]
    
    for file_path, description in config_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    print()
    
    # Validate scripts
    print("🔧 Validating scripts:")
    if not validate_scripts():
        all_checks_passed = False
    print()
    
    # Check for tests
    print("🧪 Checking test structure:")
    if not check_file_exists("tests/", "Tests directory"):
        all_checks_passed = False
    print()
    
    # Summary
    if all_checks_passed:
        print("🎉 All CI/CD validation checks passed!")
        return 0
    else:
        print("❌ Some CI/CD validation checks failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
