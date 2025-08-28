#!/usr/bin/env python3
"""
Golf AI System Startup Script
Run this to start the complete system
"""

import os
import sys
import subprocess
import time
import json
import requests
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import fastapi, uvicorn, pandas, numpy, requests, bs4
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['backend/models', 'backend/courses', 'backend/data', 'logs']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_sample_data():
    """Create sample course data"""
    courses_dir = Path("backend/courses")
    courses_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample course
    sample_course = {
        "course_id": "demo_course",
        "course_name": "Demo Golf Course",
        "holes": [
            {
                "hole_id": "demo_course_1",
                "hole_number": 1,
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
                    }
                ]
            }
        ]
    }
    
    course_file = courses_dir / "demo_course.json"
    with open(course_file, 'w') as f:
        json.dump(sample_course, f, indent=2)
    print(f"✅ Created sample course: {course_file}")

def test_backend_connection():
    """Test if backend is responding"""
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get('http://localhost:8000/health', timeout=2)
            if response.status_code == 200:
                print("✅ Backend is responding!")
                return True
        except:
            pass
        
        if i < max_attempts - 1:
            time.sleep(1)
    
    print("❌ Backend is not responding")
    return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        # Test popular courses import
        print("Testing popular courses import...")
        response = requests.post('http://localhost:8000/import-popular-courses', timeout=30)
        if response.status_code == 200:
            courses = response.json()['courses']
            print(f"✅ Imported {len(courses)} sample courses")
            
            if courses:
                # Test shot optimization
                print("Testing shot optimization...")
                test_course = courses[0]
                holes_response = requests.get(f'http://localhost:8000/courses/{test_course["course_id"]}/holes')
                
                if holes_response.status_code == 200:
                    holes = holes_response.json()
                    if holes:
                        hole_id = holes[0]['hole_id']
                        
                        shot_request = {
                            "hole_id": hole_id,
                            "distance_to_pin": 150,
                            "lateral_position": 0,
                            "shot_number": 2,
                            "lie_type": "Fairway",
                            "player_mode": "Normal"
                        }
                        
                        shot_response = requests.post('http://localhost:8000/optimize-shot', json=shot_request)
                        if shot_response.status_code == 200:
                            shot_data = shot_response.json()
                            print(f"✅ Shot optimization working: {shot_data['club']} with {shot_data['confidence']}% confidence")
                            return True
        
        print("❌ Quick test failed")
        return False
        
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

def start_backend():
    """Start the backend server"""
    print("🚀 Starting Golf AI Backend...")
    
    try:
        # Start with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error starting backend: {e}")

def main():
    """Main startup routine"""
    print("🏌️ Golf AI System Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies, run:")
        print("   pip install -r requirements.txt")
        return False
    
    # Create directories
    create_directories()
    
    # Create sample data
    create_sample_data()
    
    print("\n🌐 System will be available at:")
    print("   🔗 API: http://localhost:8000")
    print("   📚 Docs: http://localhost:8000/docs")
    print("   💓 Health: http://localhost:8000/health")
    
    print("\n📋 Quick Start Guide:")
    print("1. Import courses: POST /import-popular-courses")
    print("2. Train a hole: POST /train-hole")
    print("3. Get recommendations: POST /optimize-shot")
    
    print("\n🚀 Starting backend server...")
    print("💡 Press Ctrl+C to stop")
    print("🔄 Backend will auto-reload on file changes")
    time.sleep(3)
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()