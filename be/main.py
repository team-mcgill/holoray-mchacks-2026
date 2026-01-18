import os
import json
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
from models import VideoLabels, Label
from tracking_service import tracking_manager
import pathlib

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    tracking_manager.close_all()

app = FastAPI(title="HoloRay Medical Video Labeler with Motion Tracking", lifespan=lifespan)

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
    data = [label.model_dump() for label in payload.labels]
    
    with open(label_file, "w") as f:
        json.dump(data, f, indent=2)
        
    return {"status": "success", "count": len(payload.labels)}


# ============== TRACKING ENDPOINTS ==============

@app.post("/api/tracking/start")
def start_tracking(video_path: str, start_time: float = 0.0, annotations: List[Dict] = Body(...)):
    """
    Start a tracking session for a video with given annotations.
    """
    # Convert relative path to absolute
    full_path = os.path.join(os.getcwd(), video_path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    session_id = video_path.replace("/", "_").replace("\\", "_")
    
    success = tracking_manager.create_session(session_id, full_path, annotations, start_time)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to open video for tracking")
    
    return {
        "status": "success",
        "session_id": session_id,
        "message": "Tracking session started"
    }


@app.post("/api/tracking/frame/{session_id}")
def track_next_frame(session_id: str):
    """
    Track annotations to the next frame.
    """
    session = tracking_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Tracking session not found")
    
    result = session.track_next_frame()
    
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to track frame")
    
    return result


@app.post("/api/tracking/stop/{session_id}")
def stop_tracking(session_id: str):
    """
    Stop and close a tracking session.
    """
    tracking_manager.close_session(session_id)
    return {"status": "success", "message": "Tracking session closed"}


@app.websocket("/ws/tracking/{session_id}")
async def tracking_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time tracking updates.
    
    Client sends: { "action": "start" | "stop" | "update_annotations", ... }
    Server sends: Tracking results for each frame
    """
    await websocket.accept()
    print(f"[WS] Connection accepted for {session_id}")
    
    session = tracking_manager.get_session(session_id)
    if session is None:
        print(f"[WS] Session not found: {session_id}")
        await websocket.send_json({"error": "Session not found"})
        await websocket.close()
        return
    
    is_tracking = False
    frame_count = 0
    
    try:
        while True:
            # Check for incoming messages (non-blocking with timeout)
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.033  # ~30fps check rate
                )
                
                action = data.get("action")
                print(f"[WS] Received action: {action}")
                
                if action == "start":
                    is_tracking = True
                    last_synced_time = -1.0
                    # Immediately send initial annotation at exact drawn position
                    frame_idx = session.playback_frame
                    result = session.get_annotations_for_frame(frame_idx)
                    if result:
                        await websocket.send_json(result)
                        print(f"[WS] Tracking started, sent initial frame {frame_idx}")
                    else:
                        print(f"[WS] Tracking started (frame {frame_idx} not yet in buffer)")
                elif action == "stop":
                    is_tracking = False
                    print(f"[WS] Tracking stopped after {frame_count} frames")
                elif action == "update_annotations":
                    current_time = data.get("time", 0.0)
                    session.update_annotations(data.get("annotations", []), current_time)
                elif action == "seek":
                    frame_idx = data.get("frame", 0)
                    result = session.seek_to_frame(frame_idx)
                    if result:
                        await websocket.send_json(result)
                elif action == "sync":
                    # Sync tracking with frontend video time using predictive buffer
                    current_time = data.get("time", 0.0)
                    if is_tracking and current_time != last_synced_time:
                        frame_idx = int(current_time * session.fps)
                        result = session.get_annotations_for_frame(frame_idx)
                        if result:
                            await websocket.send_json(result)
                            frame_count += 1
                            if frame_count % 30 == 0:
                                with session.buffer_lock:
                                    buf_size = len(session.buffer)
                                print(f"[WS] Frame {frame_idx}, buffer: {buf_size} frames")
                        last_synced_time = current_time
                        
            except asyncio.TimeoutError:
                pass  # No message received, continue tracking if active
            except Exception as e:
                print(f"[WS] Error receiving message: {e}")
                break
            
            # Just wait for sync messages (no auto-streaming)
            await asyncio.sleep(0.01)
                
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected after {frame_count} frames")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
    finally:
        print(f"[WS] Connection closed for {session_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
