#!/usr/bin/env python3
"""
Pre-compute dense point trajectories for all videos using LiteTracker.
Based on official demo.py - runs on CUDA for best performance.

Usage:
    python preprocess_all.py -i ../dataset -o ../tracking_cache -s 48
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import torch
import imageio.v3 as iio

from src.model_utils import get_points_on_a_grid
from src.lite_tracker import LiteTracker

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


def preprocess_video(model, video_path: str, output_path: str, grid_size: int = 48):
    """Process a single video and save tracking data."""
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return False
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video: {width}x{height} @ {source_fps}fps, {total_frames} frames")
    print(f"Grid: {grid_size}x{grid_size} = {grid_size**2} points")
    
    # Create grid of query points
    grid_pts = get_points_on_a_grid(grid_size, (height, width))
    queries = torch.cat([
        torch.zeros_like(grid_pts[:, :, :1]),  # query frame 0
        grid_pts
    ], dim=2).to(device)
    
    # Storage
    all_coords = []
    all_visibility = []
    
    # Reset model state
    model.reset()
    
    # Process frames using imageio (same as demo.py)
    with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype, enabled=(device == "cuda")):
        for i, frame in tqdm(enumerate(iio.imiter(video_path, plugin="FFMPEG")), total=total_frames, desc="Tracking"):
            frame_tensor = torch.tensor(frame, device=device).permute(2, 0, 1)[None].float()
            
            with torch.no_grad():
                coords, vis, conf = model(frame_tensor, queries)
            
            # coords: [1, 1, N, 2], vis: [1, 1, N]
            all_coords.append(coords[0, 0].cpu().numpy())
            all_visibility.append(vis[0, 0].cpu().numpy())
    
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
        fps=source_fps,
        source_fps=source_fps,
        frame_skip=1,  # No frame skipping
        total_frames=len(all_coords)
    )
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved: {output_path} ({file_size:.1f} MB)")
    
    # Print stats
    print(f"Coords range: [{coords_array.min():.1f}, {coords_array.max():.1f}]")
    print(f"Frame 0 range: x=[{coords_array[0,:,0].min():.1f}, {coords_array[0,:,0].max():.1f}], y=[{coords_array[0,:,1].min():.1f}, {coords_array[0,:,1].max():.1f}]")
    if len(all_coords) > 100:
        print(f"Frame 100 range: x=[{coords_array[100,:,0].min():.1f}, {coords_array[100,:,0].max():.1f}], y=[{coords_array[100,:,1].min():.1f}, {coords_array[100,:,1].max():.1f}]")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-compute dense tracking for all videos")
    parser.add_argument("--input", "-i", default="../dataset", help="Input directory with videos")
    parser.add_argument("--output", "-o", default="../tracking_cache", help="Output directory for .npz files")
    parser.add_argument("--grid", "-s", type=int, default=48, help="Grid size (default: 48)")
    parser.add_argument("--weights", "-w", default="weights/scaled_online.pth", help="LiteTracker weights")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load model once
    print("Loading LiteTracker model...")
    model = LiteTracker()
    with open(args.weights, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Find all videos
    videos = []
    for ext in ["*.mp4", "*.mov", "*.avi"]:
        videos.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
    
    print(f"Found {len(videos)} videos")
    
    for video_path in videos:
        # Create output filename: dataset/Echo/echo1.mp4 -> Echo__echo1.mp4.tracking.npz
        rel_path = os.path.relpath(video_path, args.input)
        output_name = rel_path.replace(os.sep, "__").replace("/", "__") + ".tracking.npz"
        output_path = os.path.join(args.output, output_name)
        
        if os.path.exists(output_path) and not args.force:
            print(f"\nSkipping (exists): {video_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"Output: {output_path}")
        print(f"{'='*60}")
        
        try:
            preprocess_video(model, video_path, output_path, args.grid)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
