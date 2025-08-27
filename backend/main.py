"""
main.py
=======

This module defines the HTTP API for the golf strategy builder backend.  It
is built with FastAPI and provides endpoints for uploading shot data,
scraping or importing courses, training models, and obtaining shot
optimisations.  The backend relies on a simple course management system
(:mod:`course_manager`), a scorecard scraper (:mod:`scorecard_scraper`)
and a heuristic optimiser (:mod:`instant_optimizer`).  Model training is
stubbed out in this proof‑of‑concept but can be extended with a real
reinforcement learning implementation.
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from typing import Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Add src directory to sys.path for local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
# Ensure both the backend directory and the src subdirectory are on the
# Python path.  This allows sibling modules (e.g. train_model.py) and
# submodules under src (e.g. parse_trackman_csv.py) to be imported
# without requiring package installation.
for path in (CURRENT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.append(path)

from course_manager import CourseManager  # type: ignore
from instant_optimizer import InstantGolfOptimizer  # type: ignore
from parse_trackman_csv import parse_trackman_csv  # type: ignore
from scorecard_scraper import ScorecardScraper, scrape_popular_courses  # type: ignore
from train_model import train_model_for_hole, save_model, load_model  # type: ignore


# --- Configuration ---
DATA_FILE = os.path.join(CURRENT_DIR, "shots_clean.json")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
COURSES_DIR = os.path.join(CURRENT_DIR, "courses")

# Ensure persistent directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COURSES_DIR, exist_ok=True)


app = FastAPI(title="AI Golf Strategy Builder API", version="2.0.0")

# Enable CORS for all origins to facilitate local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global state ---
course_manager = CourseManager(COURSES_DIR)
active_optimizers: Dict[str, InstantGolfOptimizer] = {}


# --- Pydantic models for request bodies ---
class ShotRequest(BaseModel):
    hole_id: str
    distance_to_pin: float
    lateral_position: float = 0.0
    shot_number: int = 1
    lie_type: str = "Fairway"
    player_mode: str = "Normal"


class PositionUpdate(BaseModel):
    hole_id: str
    ball_x: float
    ball_y: float
    shot_number: int = 1


class CourseImport(BaseModel):
    course_name: str
    holes: List[Dict]


class TrainingRequest(BaseModel):
    hole_id: str
    episodes: int = 100000


# --- File upload / download ---
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)) -> Dict[str, str | int]:
    """Upload a TrackMan CSV and append its contents to the shot data store."""
    contents = await file.read()
    parsed = parse_trackman_csv(StringIO(contents.decode("utf-8")))

    # Load existing shots if available
    existing: List[Dict] = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.extend(parsed)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return {"message": f"{len(parsed)} new shots added", "total": len(existing)}


@app.get("/download")
async def download_clean_file() -> FileResponse | JSONResponse:
    """Download all recorded shot data as a JSON file."""
    if not os.path.exists(DATA_FILE):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(DATA_FILE, filename="shots_clean.json")


# --- Course scraping and import ---
@app.post("/scrape-course")
async def scrape_and_import_course(course_url: str, course_name: Optional[str] = None) -> Dict[str, Any]:
    """Scrape a course scorecard from a URL and import it as a new course."""
    try:
        scraper = ScorecardScraper()
        course_data = scraper.scrape_course(course_url, course_name)
        course_id = course_manager.save_course(course_data["course_name"], course_data["holes"])
        return {
            "message": f"Course '{course_data['course_name']}' scraped and imported successfully",
            "course_id": course_id,
            "holes_count": len(course_data["holes"]),
            "source_url": course_url,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to scrape course: {exc}")


@app.post("/import-popular-courses")
async def import_popular_courses() -> Dict[str, Any]:
    """Import a curated list of popular courses into the system."""
    courses_data = scrape_popular_courses()
    imported: List[Dict[str, Any]] = []
    for course_data in courses_data:
        try:
            course_id = course_manager.save_course(course_data["course_name"], course_data["holes"])
            imported.append(
                {
                    "course_id": course_id,
                    "course_name": course_data["course_name"],
                    "holes_count": len(course_data["holes"]),
                }
            )
        except Exception:
            # Skip courses that fail to import
            continue
    return {"message": f"Imported {len(imported)} courses", "courses": imported}


@app.post("/courses/import")
async def import_course(course_data: CourseImport) -> Dict[str, Any]:
    """Import a new course definition provided by the client."""
    try:
        course_id = course_manager.save_course(course_data.course_name, course_data.holes)
        return {
            "message": f"Course '{course_data.course_name}' imported successfully",
            "course_id": course_id,
            "holes_count": len(course_data.holes),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/courses")
async def list_courses() -> List[Dict[str, Any]]:
    """Return a list of all imported courses."""
    return course_manager.list_courses()


@app.get("/courses/{course_id}/holes")
async def get_course_holes(course_id: str) -> List[Dict[str, Any]]:
    """Return the hole definitions for a given course."""
    holes = course_manager.get_course_holes(course_id)
    if not holes:
        raise HTTPException(status_code=404, detail="Course not found")
    return holes


@app.get("/holes/{hole_id}")
async def get_hole_data(hole_id: str) -> Dict[str, Any]:
    """Return the data for a single hole."""
    hole_data = course_manager.get_hole_data(hole_id)
    if not hole_data:
        raise HTTPException(status_code=404, detail="Hole not found")
    return hole_data


# --- Training and model management ---
@app.post("/train-hole")
async def train_hole_model(request: TrainingRequest) -> Dict[str, Any]:
    """Train a model for the specified hole and make it available for optimisation."""
    hole_data = course_manager.get_hole_data(request.hole_id)
    if not hole_data:
        raise HTTPException(status_code=404, detail="Hole not found")
    # Load shot data if any
    shot_data: List[Dict] = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            shot_data = json.load(f)
    # Train model (stub)
    print(f"[train_hole_model] Starting training for hole {request.hole_id} ({request.episodes} episodes)")
    qnet, training_stats = train_model_for_hole(hole_data=hole_data, shot_data=shot_data, episodes=request.episodes)
    # Save and activate model
    model_path = save_model(qnet, request.hole_id, MODELS_DIR)
    optimizer = InstantGolfOptimizer(qnet, hole_data)
    active_optimizers[request.hole_id] = optimizer
    return {
        "message": f"Training completed for hole {request.hole_id}",
        "episodes": request.episodes,
        "model_saved": model_path,
        "training_stats": training_stats,
        "ready_for_optimization": True,
    }


@app.post("/load-model/{hole_id}")
async def load_hole_model(hole_id: str) -> Dict[str, Any]:
    """Load an existing model from disk into memory for optimisation."""
    if hole_id in active_optimizers:
        return {"message": f"Model for hole {hole_id} already loaded", "ready": True}
    hole_data = course_manager.get_hole_data(hole_id)
    if not hole_data:
        raise HTTPException(status_code=404, detail="Hole not found")
    model_path = os.path.join(MODELS_DIR, f"{hole_id}.pth")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="No trained model found. Please train first.")
    qnet = load_model(model_path)
    optimizer = InstantGolfOptimizer(qnet, hole_data)
    active_optimizers[hole_id] = optimizer
    return {"message": f"Model loaded for hole {hole_id}", "ready": True}


# --- Optimisation endpoints ---
@app.post("/optimize-shot")
async def get_optimal_shot(request: ShotRequest) -> Dict[str, Any]:
    """Return the AI's recommendation for the next shot."""
    if request.hole_id not in active_optimizers:
        # Try loading automatically
        try:
            await load_hole_model(request.hole_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Model not trained or loaded for this hole")
    optimizer = active_optimizers[request.hole_id]
    try:
        result = optimizer.get_optimal_shot(
            request.distance_to_pin,
            request.lateral_position,
            request.shot_number,
            request.lie_type,
            request.player_mode,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}")


@app.post("/reoptimize-position")
async def reoptimize_from_new_position(request: PositionUpdate) -> Dict[str, Any]:
    """Recompute the best shot after the ball has been repositioned."""
    if request.hole_id not in active_optimizers:
        raise HTTPException(status_code=400, detail="Model not loaded for this hole")
    optimizer = active_optimizers[request.hole_id]
    try:
        result = optimizer.reoptimize_from_position(request.ball_x, request.ball_y, request.shot_number)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reoptimization failed: {exc}")


@app.get("/shot-options/{hole_id}")
async def get_shot_options(
    hole_id: str, distance: float, lateral: float, shot_num: int = 1, top_n: int = 3
) -> Dict[str, Any]:
    """Return the top ``n`` shot options for comparison."""
    if hole_id not in active_optimizers:
        raise HTTPException(status_code=400, detail="Model not loaded for this hole")
    optimizer = active_optimizers[hole_id]
    try:
        options = optimizer.get_shot_options(distance, lateral, shot_num, top_n)
        return {"options": options}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get options: {exc}")


# --- Status and health endpoints ---
@app.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Return high level information about the server state."""
    return {
        "loaded_models": list(active_optimizers.keys()),
        "available_courses": len(course_manager.list_courses()),
        "shot_data_available": os.path.exists(DATA_FILE),
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy", "version": app.version}


if __name__ == "__main__":
    import uvicorn  # type: ignore

    uvicorn.run(app, host="0.0.0.0", port=8000)