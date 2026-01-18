#!/usr/bin/env python3
"""
Pre-compute dense point trajectories for all videos using LiteTracker.
Run once offline, then use for instant runtime tracking.
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lite-tracker'))

import torch
from src.lite_tracker import LiteTracker
from src.model_utils import get_points_on_a_grid


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def preprocess_video(video_path: str, output_path: str, grid_size: int = 32, weights_path: str = None, target_fps: int = 30):
    """
    Pre-compute dense tracking for a single video.
    
    Args:
        video_path: Path to input video
        output_path: Path to save .npz tracking data
        grid_size: Grid density (grid_size x grid_size points)
        weights_path: Path to LiteTracker weights
    """
    device = get_device()
    print(f"Using device: {device}")
    dtype = torch.float32
    
    # Load model
    model = LiteTracker()
    if weights_path and os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return False
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Downsample if source fps is higher than target
    frame_skip = max(1, int(round(source_fps / target_fps)))
    effective_fps = source_fps / frame_skip
    effective_frames = total_frames // frame_skip
    
    print(f"Video: {width}x{height} @ {source_fps}fps, {total_frames} frames")
    if frame_skip > 1:
        print(f"Downsampling: skip every {frame_skip} frames -> {effective_fps:.1f}fps, {effective_frames} frames")
    print(f"Grid: {grid_size}x{grid_size} = {grid_size**2} points")
    
    # Create grid of query points
    grid_pts = get_points_on_a_grid(grid_size, (height, width))
    queries = torch.cat([
        torch.zeros_like(grid_pts[:, :, :1]),  # frame 0
        grid_pts
    ], dim=2).to(device)
    
    # Storage for all frames
    all_coords = []
    all_visibility = []
    
    # Process frames
    model.reset()
    frame_idx = 0
    
    for source_frame_idx in tqdm(range(total_frames), desc="Tracking"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for downsampling
        if source_frame_idx % frame_skip != 0:
            continue
        
        # Convert frame to tensor [B, C, H, W]
        frame_rgb = frame[:, :, ::-1].copy()
        frame_tensor = torch.tensor(frame_rgb, device=device, dtype=dtype)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            coords, vis, conf = model(frame_tensor, queries)
        
        # coords: [1, 1, N, 2], vis: [1, 1, N]
        all_coords.append(coords[0, 0].cpu().numpy())
        all_visibility.append(vis[0, 0].cpu().numpy())
        frame_idx += 1
    
    cap.release()
    
    # Stack into arrays
    coords_array = np.stack(all_coords, axis=0)  # [num_frames, num_points, 2]
    vis_array = np.stack(all_visibility, axis=0)  # [num_frames, num_points]
    
    # Save
    np.savez_compressed(
        output_path,
        coords=coords_array,
        visibility=vis_array,
        grid_size=np.array([grid_size, grid_size]),
        frame_size=np.array([height, width]),
        fps=effective_fps,
        source_fps=source_fps,
        frame_skip=frame_skip,
        total_frames=len(all_coords)
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({file_size:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-compute dense tracking for videos")
    parser.add_argument("--input", "-i", help="Input video or directory", default="dataset")
    parser.add_argument("--output", "-o", help="Output directory", default="tracking_cache")
    parser.add_argument("--grid", "-g", type=int, default=32, help="Grid size (default: 32)")
    parser.add_argument("--weights", "-w", help="LiteTracker weights path", 
                        default="lite-tracker/weights/scaled_online.pth")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Find videos
    if os.path.isfile(args.input):
        videos = [args.input]
    else:
        videos = glob.glob(os.path.join(args.input, "**/*.mp4"), recursive=True)
        videos += glob.glob(os.path.join(args.input, "**/*.mov"), recursive=True)
    
    print(f"Found {len(videos)} videos")
    
    for video_path in videos:
        # Create output filename
        rel_path = os.path.relpath(video_path, args.input if os.path.isdir(args.input) else ".")
        output_name = rel_path.replace(os.sep, "__").replace("/", "__") + ".tracking.npz"
        output_path = os.path.join(args.output, output_name)
        
        if os.path.exists(output_path) and not args.force:
            print(f"Skipping (exists): {video_path}")
            continue
        
        print(f"\nProcessing: {video_path}")
        preprocess_video(video_path, output_path, args.grid, args.weights, args.fps)


if __name__ == "__main__":
    main()
