import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
from models import VideoLabels, Label
import pathlib

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DATASET_DIR = "dataset"
LABELS_DIR = "labels"

# Ensure labels directory exists
os.makedirs(LABELS_DIR, exist_ok=True)

# Mount dataset directory to serve video files
if os.path.exists(DATASET_DIR):
    app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")

def get_label_path(video_path: str) -> str:
    # Sanitize path to create a flat filename
    # e.g. "Echo/echo1.mp4" -> "Echo__echo1.mp4.json"
    safe_name = video_path.replace(os.sep, "__").replace("/", "__")
    return os.path.join(LABELS_DIR, f"{safe_name}.json")

@app.get("/api/tree")
def get_video_tree():
    """
    Returns a recursive tree structure of the dataset directory.
    """
    def build_tree(path: str) -> List[Dict[str, Any]]:
        tree = []
        try:
            # Sort dirs first, then files
            entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
            
            for entry in entries:
                if entry.name.startswith('.'):
                    continue
                    
                if entry.is_dir():
                    tree.append({
                        "name": entry.name,
                        "type": "folder",
                        "path": os.path.relpath(entry.path, "."),  # Relative to 'be'
                        "children": build_tree(entry.path)
                    })
                elif entry.is_file() and entry.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                     # Path relative to 'be' so frontend can request /dataset/...
                     # actually, the frontend expects "dataset/..." for the src
                     # os.path.relpath(entry.path, ".") gives "dataset/Echo/echo1.mp4"
                    tree.append({
                        "name": entry.name,
                        "type": "file",
                        "path": os.path.relpath(entry.path, ".")
                    })
        except PermissionError:
            pass
        return tree

    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        return []
    
    # We want the root to be the dataset folder itself content? 
    # Or return 'dataset' as the root folder.
    # User said "folders with root being the dataset path".
    # Let's return the content OF dataset dir.
    print(f"Building tree for: {DATASET_DIR}")
    return build_tree(DATASET_DIR)

@app.get("/api/labels")
def get_labels(video_path: str):
    label_file = get_label_path(video_path)
    if not os.path.exists(label_file):
        return []
    
    try:
        with open(label_file, "r") as f:
            data = json.load(f)
            # Support both old format {path: [labels]} and new format [labels]
            # Since we are moving to per-file, it should just be [labels]
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and video_path in data:
                return data[video_path]
            return []
    except json.JSONDecodeError:
        return []

@app.post("/api/labels")
def save_labels(payload: VideoLabels):
    label_file = get_label_path(payload.video_path)
    # Convert Pydantic models to dict
    data = [label.dict() for label in payload.labels]
    
    with open(label_file, "w") as f:
        json.dump(data, f, indent=2)
        
    return {"status": "success", "count": len(payload.labels)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
