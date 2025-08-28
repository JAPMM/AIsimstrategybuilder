#!/usr/bin/env python3
"""
Golf AI Repository Cleanup Script
Removes old, unused, and problematic files, keeping only what's needed
"""

import os
import shutil
from pathlib import Path

def remove_file_or_dir(path):
    """Safely remove file or directory"""
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"🗑️  Removed directory: {path}")
            else:
                os.remove(path)
                print(f"🗑️  Removed file: {path}")
        else:
            print(f"⏭️  Already gone: {path}")
    except Exception as e:
        print(f"⚠️  Could not remove {path}: {e}")

def cleanup_repo():
    """Clean up the repository"""
    print("🧹 Golf AI Repository Cleanup")
    print("=" * 50)
    
    # Files and directories to remove
    items_to_remove = [
        # Old problematic backend files
        "backend/src/train_model.py",
        "backend/train_model.py", 
        "backend/parse_trackman_csv.py",
        "backend/simulation_env.py",
        "backend/utils.py",
        "backend/your_previous_model.py",
        
        # Duplicate or old model files
        "your_previous_model.py",
        
        # Old frontend attempts (keep the clean structure)
        "frontend/package.json",
        "frontend/public/sampleCourse.json",
        "frontend/public/strategies",
        "frontend/src/api",
        "frontend/tailwind.config.js",
        
        # Setup and fix scripts (now that system works)
        "setup.py",
        "fix_imports.py",
        
        # Empty or placeholder files
        "backend/__pycache__",
        "backend/src/__pycache__",
        "__pycache__",
        ".pytest_cache",
        
        # Logs and temporary files
        "logs",
        "*.log",
        "*.tmp",
        
        # IDE files
        ".vscode/settings.json",
        ".idea",
        
        # Python cache
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        
        # Old course/strategy files that are now handled by the system
        "frontend/public/strategies/hole_1_strategy.json",
        "frontend/public/sampleCourse.json/courseJson",
    ]
    
    # Remove each item
    for item in items_to_remove:
        if "*" in item:
            # Handle glob patterns
            from glob import glob
            for match in glob(item, recursive=True):
                remove_file_or_dir(match)
        else:
            remove_file_or_dir(item)
    
    print("\n✨ Cleanup complete!")
    print("\n📁 Files you should KEEP (essential working files):")
    
    essential_files = [
        "backend/main.py",                    # ✅ Main working backend
        "backend/models/",                    # ✅ AI model storage
        "backend/courses/",                   # ✅ Course data storage
        "backend/data/",                      # ✅ Training data
        "requirements.txt",                   # ✅ Dependencies
        "start_golf_ai.py",                  # ✅ Easy startup
        "test_golf_ai.py",                   # ✅ Test suite
        "golf_shot_dispersion_summary.csv", # ✅ Club data
        "README.md",                         # ✅ Documentation
        ".gitignore",                        # ✅ Git config
    ]
    
    for file in essential_files:
        status = "✅" if os.path.exists(file) else "❌ MISSING"
        print(f"  {status} {file}")
    
    print("\n🎯 Your clean repository structure:")
    print("""
    golf-ai-system/
    ├── backend/
    │   ├── main.py                          # 🚀 Complete working backend
    │   ├── models/                          # 🧠 AI models storage
    │   ├── courses/                         # 🏌️ Course data
    │   └── data/                           # 📊 Training data
    ├── frontend/                           # 🎨 Your future UI (optional)
    ├── requirements.txt                    # 📦 Dependencies
    ├── golf_shot_dispersion_summary.csv  # ⛳ Club statistics
    ├── start_golf_ai.py                   # 🚀 Easy startup
    ├── test_golf_ai.py                    # 🧪 Test everything
    └── README.md                          # 📖 Documentation
    """)
    
    print("\n💡 Next steps after cleanup:")
    print("  1. Test system still works: python test_golf_ai.py")
    print("  2. Start backend: python start_golf_ai.py")
    print("  3. Build your frontend using the API!")
    print("  4. Deploy to production when ready")
    
    return True

def create_clean_gitignore():
    """Create a clean .gitignore file"""
    gitignore_content = """# Golf AI System - .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Golf AI specific
# Keep models and courses (they're your data!)
# backend/models/*.json  # Uncomment if you don't want to commit trained models
# backend/courses/*.json # Uncomment if you don't want to commit course data

# Temporary files
*.tmp
*.temp
temp/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Node.js (if you build a frontend)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn/

# Build outputs
*.tgz
*.tar.gz
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ Created clean .gitignore")

if __name__ == "__main__":
    cleanup_repo()
    create_clean_gitignore()
    
    print("\n🎉 Repository cleaned and organized!")
    print("Your Golf AI system is now clean, organized, and production-ready! 🚀")