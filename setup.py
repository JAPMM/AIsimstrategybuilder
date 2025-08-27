#!/usr/bin/env python3
"""
Golf AI Setup and Deployment Script
Run this to set up your golf AI system for production use.
"""

import os
import sys
import subprocess
import json
import requests
import time
from pathlib import Path

def print_step(step, message):
    print(f"\n{'='*60}")
    print(f"STEP {step}: {message}")
    print(f"{'='*60}")

def run_command(command, description):
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'pandas', 'numpy', 
        'requests', 'beautifulsoup4', 'pydantic'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        print("Installing missing packages...")
        install_cmd = f"pip install {' '.join(missing)}"
        return run_command(install_cmd, "Installing dependencies")
    else:
        print("✅ All required packages are installed")
        return True

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'backend/models',
        'backend/courses', 
        'backend/data',
        'backend/src',
        'frontend/public/strategies',
        'logs'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def test_backend_startup():
    """Test if backend starts successfully"""
    print("\n🔄 Testing backend startup...")
    
    # Start backend in background
    backend_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    time.sleep(5)
    
    try:
        # Test health endpoint
        response = requests.get('http://localhost:8000/health', timeout=10)
        if response.status_code == 200:
            print("✅ Backend is running successfully")
            backend_process.terminate()
            backend_process.wait()
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend connection failed: {e}")
    
    # Clean up
    backend_process.terminate()
    backend_process.wait()
    return False

def import_sample_course():
    """Import a sample course for testing"""
    print("\n🔄 Importing sample course...")
    
    # Sample course data
    sample_course = {
        "course_name": "Demo Golf Course",
        "holes": [
            {
                "par": 4,
                "yardage": 385,
                "green_distance": 385,
                "green_depth": 25,
                "green_width": 22,
                "fairway_width": 32,
                "elevation": 0,
                "pin_position": [373, 0],
                "zones": [
                    {
                        "type": "Bunker",
                        "x_start": 200,
                        "x_end": 230,
                        "y_start": -20,
                        "y_end": -5
                    },
                    {
                        "type": "Water",
                        "x_start": 320,
                        "x_end": 385,
                        "y_start": 18,
                        "y_end": 35
                    }
                ]
            },
            {
                "par": 3,
                "yardage": 165,
                "green_distance": 165,
                "green_depth": 28,
                "green_width": 24,
                "fairway_width": 25,
                "elevation": 15,
                "pin_position": [153, -3],
                "zones": [
                    {
                        "type": "Bunker",
                        "x_start": 140,
                        "x_end": 170,
                        "y_start": -15,
                        "y_end": 15
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            'http://localhost:8000/courses/import',
            json=sample_course,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sample course imported: {data['course_id']}")
            return data['course_id']
        else:
            print(f"❌ Failed to import sample course: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to import sample course: {e}")
        return None

def train_sample_hole(course_id):
    """Train AI on sample hole"""
    print("\n🔄 Training AI on sample hole (this will take a few minutes)...")
    
    # Get holes from course
    try:
        response = requests.get(f'http://localhost:8000/courses/{course_id}/holes')
        if response.status_code == 200:
            holes = response.json()
            if holes:
                hole_id = holes[0]['hole_id']
                
                # Start training
                train_response = requests.post(
                    'http://localhost:8000/train-hole',
                    json={
                        "hole_id": hole_id,
                        "episodes": 10000  # Reduced for demo
                    },
                    timeout=600  # 10 minutes
                )
                
                if train_response.status_code == 200:
                    print("✅ Sample hole training completed")
                    return hole_id
                else:
                    print(f"❌ Training failed: {train_response.text}")
                    
    except requests.exceptions.RequestException as e:
        print(f"❌ Training request failed: {e}")
    
    return None

def create_launch_scripts():
    """Create convenient launch scripts"""
    
    # Backend launch script
    backend_script = '''#!/bin/bash
echo "Starting Golf AI Backend..."
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
'''
    
    with open('start_backend.sh', 'w') as f:
        f.write(backend_script)
    os.chmod('start_backend.sh', 0o755)
    
    # Windows batch file
    backend_bat = '''@echo off
echo Starting Golf AI Backend...
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
'''
    
    with open('start_backend.bat', 'w') as f:
        f.write(backend_bat)
    
    print("✅ Created launch scripts: start_backend.sh and start_backend.bat")

def main():
    """Main setup routine"""
    print("🏌️ Golf AI Setup Starting...")
    print("This will set up your golf AI system for production use.")
    
    # Step 1: Check dependencies
    print_step(1, "CHECKING DEPENDENCIES")
    if not check_dependencies():
        print("❌ Setup failed at dependency check")
        return False
    
    # Step 2: Create directory structure
    print_step(2, "CREATING DIRECTORY STRUCTURE")
    create_directory_structure()
    
    # Step 3: Test backend
    print_step(3, "TESTING BACKEND")
    backend_process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', '8000'
    ])
    
    # Wait for backend to start
    time.sleep(10)
    
    if not test_backend_startup():
        backend_process.terminate()
        print("❌ Setup failed at backend test")
        return False
    
    # Step 4: Import sample course
    print_step(4, "IMPORTING SAMPLE COURSE")
    course_id = import_sample_course()
    
    if not course_id:
        backend_process.terminate()
        print("❌ Setup failed at course import")
        return False
    
    # Step 5: Train sample hole (optional)
    print_step(5, "TRAINING SAMPLE HOLE")
    response = input("Would you like to train the sample hole now? (y/n): ").lower()
    
    if response == 'y':
        hole_id = train_sample_hole(course_id)
        if hole_id:
            print(f"✅ Training completed for hole: {hole_id}")
        else:
            print("⚠️ Training failed, but you can train later through the UI")
    else:
        print("⏭️ Skipping training - you can train later through the UI")
    
    # Step 6: Create launch scripts
    print_step(6, "CREATING LAUNCH SCRIPTS")
    create_launch_scripts()
    
    # Clean up
    backend_process.terminate()
    backend_process.wait()
    
    # Final instructions
    print_step("COMPLETE", "SETUP FINISHED SUCCESSFULLY")
    print(f"""
🎉 Golf AI Setup Complete!

Next Steps:
1. Start the backend:
   • Linux/Mac: ./start_backend.sh
   • Windows: start_backend.bat
   • Manual: uvicorn backend.main:app --host 0.0.0.0 --port 8000

2. Open your browser to: http://localhost:8000/docs
   This will show the API documentation

3. Access the system:
   • API: http://localhost:8000
   • Sample course ID: {course_id}

4. Import more courses:
   • Use the /scrape-course endpoint
   • Or manually add course data via /courses/import

5. Train holes:
   • Use /train-hole endpoint for each hole
   • Training takes 5-10 minutes per hole
   • Only needs to be done once per hole

Ready to optimize your golf game! 🏌️‍♂️
""")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)