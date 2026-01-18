"""
LiteTracker Wrapper - High-performance point tracking for medical video.
7x faster than CoTracker3, designed for tissue/surgical tracking.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lite-tracker'))

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.lite_tracker import LiteTracker


@dataclass
class LiteTrackingResult:
    """Result from LiteTracker tracking."""
    boundary_points: np.ndarray  # Tracked boundary points [N, 2]
    visibility: np.ndarray       # Per-point visibility [N]
    confidence: np.ndarray       # Per-point confidence [N]
    mean_confidence: float


class LiteTrackerWrapper:
    """
    Wrapper for LiteTracker that tracks polygon vertices directly.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
        else:
            self.device = device
        
        self.dtype = torch.float32
        
        # Load model
        self.model = LiteTracker()
        
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(__file__), '..', 'lite-tracker', 
                'weights', 'scaled_online.pth'
            )
        
        if os.path.exists(weights_path):
            with open(weights_path, "rb") as f:
                state_dict = torch.load(f, map_location="cpu", weights_only=True)
                if "model" in state_dict:
                    state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)
            print(f"[LiteTracker] Loaded weights from {weights_path}")
        else:
            print(f"[LiteTracker] WARNING: No weights found at {weights_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # State
        self.annotations: Dict[str, dict] = {}
        self.queries: Optional[torch.Tensor] = None
        self.frame_idx = 0
        self.frame_shape: Optional[Tuple[int, int]] = None
    
    def initialize(
        self,
        frame: np.ndarray,
        annotations: List[Dict]
    ) -> None:
        """
        Initialize tracker with first frame and annotations.
        
        Args:
            frame: First video frame (BGR, HxWxC)
            annotations: List of annotation dicts with 'id' and 'points'
        """
        self.model.reset()
        self.frame_idx = 0
        
        h, w = frame.shape[:2]
        self.frame_shape = (h, w)
        
        # Collect all points from all annotations
        all_points = []
        point_to_ann = []  # Maps each point to its annotation
        
        self.annotations = {}
        
        for ann in annotations:
            ann_id = ann['id']
            
            if 'points' in ann and ann['points'] and len(ann['points']) > 2:
                boundary_pct = np.array(ann['points'], dtype=np.float32)
                boundary_px = boundary_pct.copy()
                boundary_px[:, 0] = boundary_px[:, 0] * w / 100.0
                boundary_px[:, 1] = boundary_px[:, 1] * h / 100.0
            else:
                x = ann['x'] * w / 100.0
                y = ann['y'] * h / 100.0
                bw = ann['width'] * w / 100.0
                bh = ann['height'] * h / 100.0
                boundary_px = np.array([
                    [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
                ], dtype=np.float32)
            
            start_idx = len(all_points)
            for pt in boundary_px:
                all_points.append(pt)
                point_to_ann.append(ann_id)
            end_idx = len(all_points)
            
            self.annotations[ann_id] = {
                'point_indices': (start_idx, end_idx),
                'boundary_px': boundary_px,
                'meta': ann
            }
        
        if not all_points:
            print("[LiteTracker] No points to track")
            return
        
        # Build queries tensor: [B, N, 3] where 3 = (frame_idx, x, y)
        all_points = np.array(all_points, dtype=np.float32)
        N = len(all_points)
        
        # LiteTracker expects (frame_idx, x, y)
        queries = np.zeros((1, N, 3), dtype=np.float32)
        queries[0, :, 0] = 0  # All points queried at frame 0
        queries[0, :, 1] = all_points[:, 0]  # x
        queries[0, :, 2] = all_points[:, 1]  # y
        
        self.queries = torch.tensor(queries, device=self.device, dtype=self.dtype)
        
        # Process first frame
        frame_tensor = self._frame_to_tensor(frame)
        
        with torch.no_grad():
            coords, vis, conf = self.model(frame_tensor, self.queries)
        
        # Store initial results
        coords_np = coords[0, 0].cpu().numpy()  # [N, 2]
        
        for ann_id, data in self.annotations.items():
            start, end = data['point_indices']
            data['boundary_px'] = coords_np[start:end]
        
        self.frame_idx = 1
        print(f"[LiteTracker] Initialized {len(annotations)} annotations, {N} points on {self.device}")
    
    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert BGR frame to tensor [B, C, H, W]."""
        if len(frame.shape) == 2:
            frame = np.stack([frame, frame, frame], axis=-1)
        
        # BGR to RGB
        frame_rgb = frame[:, :, ::-1].copy()
        
        # [H, W, C] -> [B, C, H, W]
        frame_tensor = torch.tensor(frame_rgb, device=self.device, dtype=self.dtype)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor
    
    def track_frame(self, frame: np.ndarray) -> Dict[str, LiteTrackingResult]:
        """
        Track all annotations to new frame.
        
        Args:
            frame: New video frame (BGR)
        
        Returns:
            Dict mapping annotation ID to LiteTrackingResult
        """
        if self.queries is None or not self.annotations:
            return {}
        
        frame_tensor = self._frame_to_tensor(frame)
        
        with torch.no_grad():
            coords, vis, conf = self.model(frame_tensor, self.queries)
        
        # coords: [B, T, N, 2], vis: [B, T, N], conf: [B, T, N]
        coords_np = coords[0, 0].cpu().numpy()  # [N, 2]
        vis_np = vis[0, 0].cpu().numpy()        # [N]
        conf_np = conf[0, 0].cpu().numpy()      # [N]
        
        results = {}
        
        for ann_id, data in self.annotations.items():
            start, end = data['point_indices']
            
            pts = coords_np[start:end]
            v = vis_np[start:end]
            c = conf_np[start:end]
            
            data['boundary_px'] = pts
            
            results[ann_id] = LiteTrackingResult(
                boundary_points=pts.astype(np.float32),
                visibility=v.astype(np.float32),
                confidence=c.astype(np.float32),
                mean_confidence=float(c.mean()) if len(c) > 0 else 0.0
            )
        
        self.frame_idx += 1
        return results
    
    def get_annotations_dict(
        self,
        results: Dict[str, LiteTrackingResult],
        frame_shape: Tuple[int, int]
    ) -> List[Dict]:
        """Convert tracking results to annotation dicts for frontend."""
        h, w = frame_shape[:2]
        output = []
        
        for ann_id, result in results.items():
            meta = self.annotations[ann_id]['meta']
            
            # Convert boundary back to percentage
            boundary_pct = result.boundary_points.copy()
            boundary_pct[:, 0] = boundary_pct[:, 0] * 100.0 / w
            boundary_pct[:, 1] = boundary_pct[:, 1] * 100.0 / h
            
            # Compute bounding box
            x_min, y_min = boundary_pct.min(axis=0)
            x_max, y_max = boundary_pct.max(axis=0)
            
            output.append({
                'id': ann_id,
                'label': meta.get('label', 'Object'),
                'color': meta.get('color', '#0ea5e9'),
                'x': float(x_min),
                'y': float(y_min),
                'width': float(x_max - x_min),
                'height': float(y_max - y_min),
                'points': boundary_pct.tolist(),
                'confidence': result.mean_confidence,
                'visibility': result.visibility.tolist()
            })
        
        return output
    
    def reset(self):
        """Reset tracker for new video."""
        self.model.reset()
        self.annotations = {}
        self.queries = None
        self.frame_idx = 0
