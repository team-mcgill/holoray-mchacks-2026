"""
PrecomputedTracker - Uses pre-computed dense point trajectories for instant tracking.
Interpolates user-drawn annotation points from nearest grid points.
"""
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import interpolate


@dataclass
class PrecomputedTrackingResult:
    """Result from precomputed tracking."""
    boundary_points: np.ndarray  # [N, 2]
    visibility: np.ndarray       # [N]
    confidence: float


# Global cache for loaded .npz data (shared across tracker instances)
_loaded_data_cache: Dict[str, dict] = {}

class PrecomputedTracker:
    """
    Uses pre-computed dense tracking data for instant runtime tracking.
    Interpolates motion from nearest grid points to user-drawn points.
    """
    
    def __init__(self, cache_dir: str = "tracking_cache"):
        self.cache_dir = cache_dir
        self.data: Optional[dict] = None
        self.video_path: Optional[str] = None
        self.annotations: Dict[str, dict] = {}
        self.frame_size: Tuple[int, int] = (0, 0)
        self.current_frame: int = 0
        self.start_frame: int = 0  # Frame where annotations were drawn
        self.frame_skip: int = 1
        self.source_fps: float = 30.0
        self.effective_fps: float = 30.0
    
    def _get_cache_path(self, video_path: str) -> str:
        """Get cache file path for a video."""
        # Extract relative path from dataset folder
        # video_path could be: /full/path/be/dataset/Echo/echo1.mp4 or dataset/Echo/echo1.mp4
        if 'dataset' in video_path:
            idx = video_path.find('dataset')
            rel_path = video_path[idx + len('dataset') + 1:]  # e.g., "Echo/echo1.mp4"
        else:
            rel_path = os.path.basename(video_path)
        
        # Match naming convention from preprocess_tracking.py
        cache_name = rel_path.replace(os.sep, "__").replace("/", "__") + ".tracking.npz"
        return os.path.join(self.cache_dir, cache_name)
    
    def has_precomputed_data(self, video_path: str) -> bool:
        """Check if pre-computed tracking exists for video."""
        cache_path = self._get_cache_path(video_path)
        return os.path.exists(cache_path)
    
    def load(self, video_path: str) -> bool:
        """Load pre-computed tracking data for video (uses global cache)."""
        global _loaded_data_cache
        cache_path = self._get_cache_path(video_path)
        
        # Check global cache first
        if cache_path in _loaded_data_cache:
            self.data = _loaded_data_cache[cache_path]
            self.video_path = video_path
            self.frame_size = tuple(self.data['frame_size'])
            self.frame_skip = int(self.data.get('frame_skip', 1))
            self.source_fps = float(self.data.get('source_fps', 30))
            self.effective_fps = float(self.data.get('fps', 30))
            print(f"[PrecomputedTracker] Using cached data for {cache_path}")
            return True
        
        if not os.path.exists(cache_path):
            print(f"[PrecomputedTracker] No cache found: {cache_path}")
            return False
        
        try:
            self.data = dict(np.load(cache_path))
            self.video_path = video_path
            self.frame_size = tuple(self.data['frame_size'])
            self.frame_skip = int(self.data.get('frame_skip', 1))
            self.source_fps = float(self.data.get('source_fps', 30))
            self.effective_fps = float(self.data.get('fps', 30))
            
            # Store in global cache
            _loaded_data_cache[cache_path] = self.data
            
            print(f"[PrecomputedTracker] Loaded {cache_path}")
            print(f"  - Frames: {self.data['total_frames']}")
            print(f"  - Grid: {self.data['grid_size']}")
            print(f"  - Size: {self.frame_size}")
            print(f"  - FPS: {self.source_fps} -> {self.effective_fps} (skip={self.frame_skip})")
            return True
        except Exception as e:
            print(f"[PrecomputedTracker] Failed to load: {e}")
            return False
    
    def initialize(self, frame: np.ndarray, annotations: List[Dict], start_frame: int = 0) -> None:
        """Initialize with annotations at given start frame."""
        h, w = frame.shape[:2]
        self.frame_size = (h, w)
        self.current_frame = start_frame
        self.start_frame = start_frame
        self.annotations = {}
        
        for ann in annotations:
            ann_id = ann['id']
            
            if 'points' in ann and ann['points'] and len(ann['points']) > 2:
                boundary_pct = np.array(ann['points'], dtype=np.float32)
                
                # Check if points are in percentage (0-100) or already pixels
                # Valid percentage coords should be in 0-100 range
                max_val = np.max(np.abs(boundary_pct))
                if max_val <= 100:
                    # Points are in percentage, convert to pixels
                    boundary_px = boundary_pct.copy()
                    boundary_px[:, 0] = boundary_px[:, 0] * w / 100.0
                    boundary_px[:, 1] = boundary_px[:, 1] * h / 100.0
                else:
                    # Points are already in some other format, likely corrupted
                    # Fall back to using x/y/width/height
                    print(f"[PrecomputedTracker] WARNING: points out of range (max={max_val}), using bbox")
                    x = ann.get('x', 50) * w / 100.0
                    y = ann.get('y', 50) * h / 100.0
                    bw = ann.get('width', 10) * w / 100.0
                    bh = ann.get('height', 10) * h / 100.0
                    boundary_px = np.array([
                        [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
                    ], dtype=np.float32)
            else:
                # x/y/width/height should be in percentage (0-100)
                x_pct = ann.get('x', 50)
                y_pct = ann.get('y', 50)
                w_pct = ann.get('width', 10)
                h_pct = ann.get('height', 10)
                
                # Validate they're in percentage range
                if max(abs(x_pct), abs(y_pct), abs(w_pct), abs(h_pct)) > 100:
                    print(f"[PrecomputedTracker] WARNING: bbox out of range, using defaults")
                    x_pct, y_pct, w_pct, h_pct = 45, 45, 10, 10
                
                x = x_pct * w / 100.0
                y = y_pct * h / 100.0
                bw = w_pct * w / 100.0
                bh = h_pct * h / 100.0
                boundary_px = np.array([
                    [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
                ], dtype=np.float32)
            
            self.annotations[ann_id] = {
                'initial_points': boundary_px.copy(),
                'current_points': boundary_px.copy(),
                'meta': ann
            }
        
        print(f"[PrecomputedTracker] Initialized {len(annotations)} annotations at start_frame={start_frame}")
        print(f"[PrecomputedTracker] frame_size={self.frame_size}, frame_skip={self.frame_skip}")
        for ann_id, data in self.annotations.items():
            print(f"[PrecomputedTracker] Annotation {ann_id}: initial_points[0]={data['initial_points'][0]}")
    
    def _video_frame_to_cache_frame(self, video_frame_idx: int) -> int:
        """Convert video frame index to cache frame index (accounting for frame_skip)."""
        return video_frame_idx // self.frame_skip
    
    def _interpolate_point(self, point: np.ndarray, video_frame_idx: int) -> Tuple[np.ndarray, float]:
        """
        Interpolate a single point's position at given frame using nearest neighbor
        from the closest grid point's motion.
        
        Args:
            point: [x, y] in pixel coordinates at start_frame
            video_frame_idx: Target frame index in video (will be mapped to cache frame)
        
        Returns:
            (new_point, visibility)
        """
        if self.data is None:
            return point, 1.0
        
        coords = self.data['coords']  # [num_frames, num_points, 2]
        visibility = self.data['visibility']  # [num_frames, num_points]
        
        num_frames = coords.shape[0]
        
        # Map video frames to cache frames
        cache_frame_idx = self._video_frame_to_cache_frame(video_frame_idx)
        cache_frame_idx = min(max(0, cache_frame_idx), num_frames - 1)
        
        start_cache_frame = self._video_frame_to_cache_frame(self.start_frame)
        start_cache_frame = min(max(0, start_cache_frame), num_frames - 1)
        
        # Get grid coordinates at start frame (where annotation was drawn)
        coords_start = coords[start_cache_frame]  # [N, 2]
        
        # Find nearest grid point by distance to where user drew
        distances = np.sum((coords_start - point) ** 2, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Get displacement from start_frame to target frame
        raw_displacement = coords[cache_frame_idx, nearest_idx] - coords[start_cache_frame, nearest_idx]
        
        # Clamp displacement to reasonable bounds (max 50% of frame per direction)
        # This prevents runaway tracking when LiteTracker loses track
        h, w = self.frame_size
        max_disp = max(h, w) * 0.5
        displacement = np.clip(raw_displacement, -max_disp, max_disp)
        

        
        # Apply displacement
        new_point = point + displacement
        
        # Also clamp final point to stay within frame (with small margin)
        margin = 0.05 * max(h, w)
        new_point[0] = np.clip(new_point[0], -margin, w + margin)
        new_point[1] = np.clip(new_point[1], -margin, h + margin)
        
        # Get visibility
        vis = float(visibility[cache_frame_idx, nearest_idx])
        
        return new_point.astype(np.float32), vis
    
    def track_frame(self, frame: np.ndarray) -> Dict[str, PrecomputedTrackingResult]:
        """Track all annotations to current frame."""
        self.current_frame += 1
        results = {}
        
        for ann_id, data in self.annotations.items():
            initial_pts = data['initial_points']
            new_pts = []
            visibilities = []
            
            for pt in initial_pts:
                new_pt, vis = self._interpolate_point(pt, self.current_frame)
                new_pts.append(new_pt)
                visibilities.append(vis)
            
            new_pts = np.array(new_pts, dtype=np.float32)
            vis_arr = np.array(visibilities, dtype=np.float32)
            
            data['current_points'] = new_pts
            
            results[ann_id] = PrecomputedTrackingResult(
                boundary_points=new_pts,
                visibility=vis_arr,
                confidence=float(vis_arr.mean())
            )
        
        return results
    
    def get_annotations_for_frame(self, frame_idx: int) -> Dict[str, PrecomputedTrackingResult]:
        """Get annotations at specific frame (O(1) with interpolation)."""
        results = {}
        
        for ann_id, data in self.annotations.items():
            initial_pts = data['initial_points']
            new_pts = []
            visibilities = []
            
            for i, pt in enumerate(initial_pts):
                new_pt, vis = self._interpolate_point(pt, frame_idx)
                new_pts.append(new_pt)
                visibilities.append(vis)
            
            new_pts = np.array(new_pts, dtype=np.float32)
            vis_arr = np.array(visibilities, dtype=np.float32)
            
            results[ann_id] = PrecomputedTrackingResult(
                boundary_points=new_pts,
                visibility=vis_arr,
                confidence=float(vis_arr.mean())
            )
        
        return results
    
    def get_annotations_dict(
        self,
        results: Dict[str, PrecomputedTrackingResult],
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
                'confidence': result.confidence,
                'visibility': result.visibility.tolist()
            })
        
        return output
    
    def reset(self):
        """Reset tracker state."""
        self.annotations = {}
        self.current_frame = 0
