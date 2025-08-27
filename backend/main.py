from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, torch
from io import StringIO
from typing import Dict, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parse_trackman_csv import parse_trackman_csv
from src.train_model import train_model_for_hole, save_model, load_model
from src.instant_optimizer import InstantGolfOptimizer
from src.course_manager import CourseManager

DATA_FILE = "backend/shots_clean.json"
MODELS_DIR = "backend/models"
COURSES_DIR = "backend/courses"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COURSES_DIR, exist_ok=True)

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
course_manager = CourseManager(COURSES_DIR)
active_optimizers: Dict[str, InstantGolfOptimizer] = {}

# Request/Response Models
class ShotRequest(BaseModel):
    hole_id: str
    distance_to_pin: float
    lateral_position: float
    shot_number: int = 1
    lie_type: str = "Fairway"
    player_mode: str = "Normal"

class PositionUpdate(BaseModel):
    hole_id: str
    ball_x: float
    ball_y: float
    shot_number: int

class CourseImport(BaseModel):
    course_name: str
    holes: List[Dict]  # Array of hole data

class TrainingRequest(BaseModel):
    hole_id: str
    episodes: int = 100000

# Existing endpoints (unchanged)
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    parsed = parse_trackman_csv(StringIO(contents.decode("utf-8")))

    # Load existing file or create new
    existing = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            existing = json.load(f)

    existing.extend(parsed)
    with open(DATA_FILE, "w") as f:
        json.dump(existing, f, indent=2)

    return {"message": f"{len(parsed)} new shots added", "total": len(existing)}

@app.get("/download")
async def download_clean_file():
    if not os.path.exists(DATA_FILE):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(DATA_FILE, filename="shots_clean.json")

# Add course scraping endpoint
@app.post("/scrape-course")
async def scrape_and_import_course(course_url: str, course_name: str = None):
    """Scrape course data from website and import"""
    try:
        from src.scorecard_scraper import ScorecardScraper
        
        scraper = ScorecardScraper()
        course_data = scraper.scrape_course(course_url, course_name)
        
        # Import the scraped course
        course_id = course_manager.save_course(course_data["course_name"], course_data["holes"])
        
        return {
            "message": f"Course '{course_data['course_name']}' scraped and imported successfully",
            "course_id": course_id,
            "holes_count": len(course_data["holes"]),
            "source_url": course_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape course: {str(e)}")

@app.post("/import-popular-courses")
async def import_popular_courses():
    """Import a set of popular courses for quick setup"""
    try:
        from src.scorecard_scraper import scrape_popular_courses
        
        courses_data = scrape_popular_courses()
        imported_courses = []
        
        for course_data in courses_data:
            try:
                course_id = course_manager.save_course(course_data["course_name"], course_data["holes"])
                imported_courses.append({
                    "course_id": course_id,
                    "course_name": course_data["course_name"],
                    "holes_count": len(course_data["holes"])
                })
            except Exception as e:
                print(f"Failed to import {course_data.get('course_name', 'unknown')}: {e}")
                continue
        
        return {
            "message": f"Imported {len(imported_courses)} popular courses",
            "courses": imported_courses
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import popular courses: {str(e)}")

# Enhanced course import with validation
class CourseImportAdvanced(BaseModel):
    course_name: str
    course_url: str = None
    holes: List[Dict]
    source_type: str = "manual"  # "manual", "scraped", "api"
@app.post("/courses/import")
async def import_course(course_data: CourseImport):
    """Import a new course with hole layouts"""
    try:
        course_id = course_manager.save_course(course_data.course_name, course_data.holes)
        return {
            "message": f"Course '{course_data.course_name}' imported successfully",
            "course_id": course_id,
            "holes_count": len(course_data.holes)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/courses")
async def list_courses():
    """Get all available courses"""
    return course_manager.list_courses()

@app.get("/courses/{course_id}/holes")
async def get_course_holes(course_id: str):
    """Get all holes for a course"""
    holes = course_manager.get_course_holes(course_id)
    if not holes:
        raise HTTPException(status_code=404, detail="Course not found")
    return holes

@app.get("/holes/{hole_id}")
async def get_hole_data(hole_id: str):
    """Get specific hole layout data"""
    hole_data = course_manager.get_hole_data(hole_id)
    if not hole_data:
        raise HTTPException(status_code=404, detail="Hole not found")
    return hole_data

# Training Endpoints
@app.post("/train-hole")
async def train_hole_model(request: TrainingRequest):
    """Train AI model for specific hole (100k episodes)"""
    try:
        # Get hole data
        hole_data = course_manager.get_hole_data(request.hole_id)
        if not hole_data:
            raise HTTPException(status_code=404, detail="Hole not found")
        
        # Load existing shot data if available
        shot_data = []
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE) as f:
                shot_data = json.load(f)
        
        # Train model
        print(f"Starting training for hole {request.hole_id} ({request.episodes} episodes)")
        trained_qnet, training_stats = train_model_for_hole(
            hole_data=hole_data,
            shot_data=shot_data,
            episodes=request.episodes
        )
        
        # Save trained model
        model_path = save_model(trained_qnet, request.hole_id, MODELS_DIR)
        
        # Load into active optimizers for instant use
        optimizer = InstantGolfOptimizer(trained_qnet, hole_data)
        active_optimizers[request.hole_id] = optimizer
        
        return {
            "message": f"Training completed for hole {request.hole_id}",
            "episodes": request.episodes,
            "model_saved": model_path,
            "training_stats": training_stats,
            "ready_for_optimization": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/load-model/{hole_id}")
async def load_hole_model(hole_id: str):
    """Load pre-trained model for instant optimization"""
    try:
        # Check if already loaded
        if hole_id in active_optimizers:
            return {"message": f"Model for hole {hole_id} already loaded", "ready": True}
        
        # Get hole data
        hole_data = course_manager.get_hole_data(hole_id)
        if not hole_data:
            raise HTTPException(status_code=404, detail="Hole not found")
        
        # Load trained model
        model_path = os.path.join(MODELS_DIR, f"{hole_id}.pth")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="No trained model found. Please train first.")
        
        trained_qnet = load_model(model_path)
        optimizer = InstantGolfOptimizer(trained_qnet, hole_data)
        active_optimizers[hole_id] = optimizer
        
        return {
            "message": f"Model loaded for hole {hole_id}",
            "ready": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Instant Optimization Endpoints
@app.post("/optimize-shot")
async def get_optimal_shot(request: ShotRequest):
    """Get instant optimal shot recommendation"""
    if request.hole_id not in active_optimizers:
        # Try to load model automatically
        try:
            await load_hole_model(request.hole_id)
        except:
            raise HTTPException(status_code=400, detail="Model not trained or loaded for this hole")
    
    optimizer = active_optimizers[request.hole_id]
    
    try:
        result = optimizer.get_optimal_shot(
            request.distance_to_pin,
            request.lateral_position,
            request.shot_number,
            request.lie_type,
            request.player_mode
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/reoptimize-position")  
async def reoptimize_from_new_position(request: PositionUpdate):
    """Instantly reoptimize when ball is moved (drag & drop)"""
    if request.hole_id not in active_optimizers:
        raise HTTPException(status_code=400, detail="Model not loaded for this hole")
    
    optimizer = active_optimizers[request.hole_id]
    
    try:
        result = optimizer.reoptimize_from_position(
            request.ball_x,
            request.ball_y, 
            request.shot_number
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reoptimization failed: {str(e)}")

@app.get("/shot-options/{hole_id}")
async def get_shot_options(hole_id: str, distance: float, lateral: float, shot_num: int = 1, top_n: int = 3):
    """Get top N shot options for comparison"""
    if hole_id not in active_optimizers:
        raise HTTPException(status_code=400, detail="Model not loaded for this hole")
    
    optimizer = active_optimizers[hole_id]
    
    try:
        options = optimizer.get_shot_options(distance, lateral, shot_num, top_n)
        return {"options": options}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get options: {str(e)}")

# Status and Health Endpoints
@app.get("/status")
async def get_system_status():
    """Get system status and loaded models"""
    return {
        "loaded_models": list(active_optimizers.keys()),
        "available_courses": len(course_manager.list_courses()),
        "shot_data_available": os.path.exists(DATA_FILE)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)