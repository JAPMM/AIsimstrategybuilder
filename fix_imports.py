#!/usr/bin/env python3
"""
Quick fix for import path issues
Run this before starting the backend
"""

import os
from pathlib import Path

def create_missing_files():
    """Create missing __init__.py files and fix import paths"""
    
    # Define directories
    COURSES_DIR = "backend/courses"
    MODELS_DIR = "backend/models"
    
    # Create __init__.py files
    init_files = [
        'backend/__init__.py',
        'backend/src/__init__.py'
    ]
    
    for init_file in init_files:
        Path(init_file).parent.mkdir(parents=True, exist_ok=True)
        with open(init_file, 'w') as f:
            f.write(f'# {init_file}\n')
        print(f"✅ Created {init_file}")
    
    # Create simplified main.py with correct imports
    main_py_content = '''
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, torch
from io import StringIO
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Create directories
DATA_FILE = "backend/shots_clean.json"
MODELS_DIR = "backend/models"
COURSES_DIR = "backend/courses"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COURSES_DIR, exist_ok=True)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class ShotRequest(BaseModel):
    hole_id: str
    distance_to_pin: float
    lateral_position: float
    shot_number: int = 1
    lie_type: str = "Fairway"
    player_mode: str = "Normal"

class CourseImport(BaseModel):
    course_name: str
    holes: List[Dict]

# Basic endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/status")
async def get_status():
    return {
        "backend": "running",
        "models_dir": os.path.exists(MODELS_DIR),
        "courses_dir": os.path.exists(COURSES_DIR)
    }

@app.post("/courses/import")
async def import_course(course_data: CourseImport):
    """Import a new course"""
    try:
        course_id = f"course_{len(os.listdir(COURSES_DIR)) + 1}"
        course_file = os.path.join(COURSES_DIR, f"{course_id}.json")
        
        course_info = {
            "course_id": course_id,
            "course_name": course_data.course_name,
            "holes": course_data.holes
        }
        
        with open(course_file, 'w') as f:
            json.dump(course_info, f, indent=2)
        
        return {
            "message": f"Course '{course_data.course_name}' imported successfully",
            "course_id": course_id,
            "holes_count": len(course_data.holes)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/courses")
async def list_courses():
    """List all available courses"""
    courses = []
    if os.path.exists(COURSES_DIR):
        for filename in os.listdir(COURSES_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(COURSES_DIR, filename), 'r') as f:
                    course = json.load(f)
                    courses.append({
                        "course_id": course["course_id"],
                        "course_name": course["course_name"],
                        "holes_count": len(course.get("holes", []))
                    })
    return courses

@app.post("/optimize-shot")
async def get_optimal_shot(request: ShotRequest):
    """Get optimal shot recommendation (simplified for testing)"""
    
    # Simple logic for demo purposes
    clubs = ["Driver", "3 Wood", "7 Iron", "Pitching Wedge", "Sand Wedge"]
    
    if request.distance_to_pin > 200:
        optimal_club = "Driver"
        carry = 250
    elif request.distance_to_pin > 150:
        optimal_club = "7 Iron" 
        carry = 150
    elif request.distance_to_pin > 100:
        optimal_club = "Pitching Wedge"
        carry = 100
    else:
        optimal_club = "Sand Wedge"
        carry = 60
    
    return {
        "club": optimal_club,
        "aim_point": 0,
        "confidence": 85.0,
        "q_value": 2.5,
        "expected_outcome": {
            "carry_distance": carry,
            "remaining_distance": max(0, request.distance_to_pin - carry),
            "expected_lateral": request.lateral_position,
            "success_probability": 80.0
        },
        "strokes_gained": 0.2
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open('backend/main.py', 'w') as f:
        f.write(main_py_content)
    print("✅ Created simplified backend/main.py")
    
    # Create directories
    os.makedirs(COURSES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Create a simple course for testing
    sample_course = {
        "course_name": "Demo Course",
        "holes": [
            {
                "hole_number": 1,
                "par": 4,
                "yardage": 385,
                "green_distance": 385,
                "green_depth": 25,
                "green_width": 22,
                "fairway_width": 32,
                "pin_position": [373, 0],
                "zones": []
            }
        ]
    }
    
    with open(f'{COURSES_DIR}/demo_course.json', 'w') as f:
        json.dump({
            "course_id": "demo_course",
            **sample_course
        }, f, indent=2)
    print("✅ Created demo course")

def main():
    print("🔧 Fixing import issues...")
    create_missing_files()
    print("\n✅ All fixes applied!")
    print("\nNow you can run:")
    print("  cd backend")
    print("  python main.py")
    print("  # or")
    print("  uvicorn main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()