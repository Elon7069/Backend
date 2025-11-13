#!/usr/bin/env python3
"""
Security Check Script - Run this before pushing to verify your API key is secure.
"""

import os
import sys
import re
from pathlib import Path

def check_env_file():
    """Check if .env file exists and is in .gitignore."""
    print("Checking .env file security...")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    gitignore_path = Path(".gitignore")
    
    # Check if .env exists
    if env_path.exists():
        print("  [OK] .env file exists (this is OK - it should be local only)")
    else:
        print("  [WARNING] .env file not found (create it from .env.example)")
    
    # Check if .env.example exists
    if env_example_path.exists():
        print("  [OK] .env.example exists (template file)")
    else:
        print("  [WARNING] .env.example not found (should exist as a template)")
    
    # Check .gitignore
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore_content = f.read()
        
        if '.env' in gitignore_content and '*.env' not in gitignore_content:
            print("  [OK] .env is properly ignored in .gitignore")
        elif '*.env' in gitignore_content:
            print("  [WARNING] *.env in .gitignore might ignore .env.example too!")
            print("     Make sure .env.example is NOT ignored")
        else:
            print("  [ERROR] .env is NOT in .gitignore!")
            return False
    else:
        print("  [ERROR] .gitignore file not found!")
        return False
    
    return True

def check_hardcoded_keys():
    """Check for hardcoded API keys in source files."""
    print("\nChecking for hardcoded API keys in code...")
    
    # Patterns that might indicate hardcoded keys
    patterns = [
        r'OPENAI_API_KEY\s*=\s*["\']sk-[a-zA-Z0-9]{20,}',
        r'api_key\s*=\s*["\']sk-[a-zA-Z0-9]{20,}',
        r'["\']sk-[a-zA-Z0-9]{48,}["\']',  # Full API key pattern
    ]
    
    issues_found = False
    
    # Check Python files
    for py_file in Path('.').glob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Check if it's just an example/placeholder
                            if 'your-actual-key' in line or 'your_key_here' in line.lower():
                                continue
                            print(f"  [ERROR] Potential hardcoded key found in {py_file.name}:{i}")
                            print(f"     {line.strip()[:80]}")
                            issues_found = True
        except Exception as e:
            print(f"  ⚠ Could not check {py_file.name}: {e}")
    
    if not issues_found:
        print("  [OK] No hardcoded API keys found in source files")
    
    return not issues_found

def check_git_status():
    """Check if .env is tracked by git."""
    print("\nChecking Git status...")
    
    try:
        import subprocess
        
        # Check if we're in a git repo
        try:
            subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True,
                check=True,
                cwd='.'
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  [WARNING] Not in a Git repository (skipping Git check)")
            return True
        
        # Check git status for .env files
        result = subprocess.run(
            ['git', 'status', '--porcelain', '.env'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        # Also check if .env is in git index
        ls_result = subprocess.run(
            ['git', 'ls-files', '.env'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        # Check if .env appears in status or is tracked
        env_in_status = '.env' in result.stdout and result.stdout.strip()
        env_tracked = '.env' in ls_result.stdout and ls_result.stdout.strip()
        
        if env_in_status or env_tracked:
            print("  [ERROR] WARNING: .env file appears to be tracked by Git!")
            print("     Run: git rm --cached .env")
            print("     Or if in Backend folder: git rm --cached Backend/.env")
            return False
        else:
            print("  [OK] .env is not tracked by Git")
            return True
    except FileNotFoundError:
        print("  [WARNING] Git not found (skipping Git check)")
        return True
    except Exception as e:
        print(f"  [WARNING] Could not check Git status: {e}")
        return True

def main():
    """Run all security checks."""
    print("=" * 60)
    print("Security Check - Pre-Push Verification")
    print("=" * 60)
    print()
    
    all_checks_passed = True
    
    # Run checks
    all_checks_passed &= check_env_file()
    all_checks_passed &= check_hardcoded_keys()
    all_checks_passed &= check_git_status()
    
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("[OK] All security checks passed! Safe to push.")
    else:
        print("[ERROR] Security issues found! Fix them before pushing.")
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()

