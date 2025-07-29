from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, json
from io import StringIO
from src.parse_trackman_csv import parse_trackman_csv
from src.train_model import train_model

DATA_FILE = "backend/shots_clean.json"

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/train")
async def trigger_training():
    if not os.path.exists(DATA_FILE):
        return JSONResponse({"error": "No shots available"}, status_code=404)

    with open(DATA_FILE) as f:
        shot_data = json.load(f)

    result = train_model(shot_data)
    return {"message": "Training finished", "summary": result}
