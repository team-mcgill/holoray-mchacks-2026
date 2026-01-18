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
    
    def _get_cache_path(self, video_path: str) -> str:
        """Get cache file path for a video."""
        # Match the naming convention from preprocess_tracking.py
        rel_path = os.path.basename(video_path)
        # Try to find in cache dir
        cache_name = rel_path.replace(os.sep, "__").replace("/", "__") + ".tracking.npz"
        return os.path.join(self.cache_dir, cache_name)
    
    def has_precomputed_data(self, video_path: str) -> bool:
        """Check if pre-computed tracking exists for video."""
        cache_path = self._get_cache_path(video_path)
        return os.path.exists(cache_path)
    
    def load(self, video_path: str) -> bool:
        """Load pre-computed tracking data for video."""
        cache_path = self._get_cache_path(video_path)
        
        if not os.path.exists(cache_path):
            print(f"[PrecomputedTracker] No cache found: {cache_path}")
            return False
        
        try:
            self.data = dict(np.load(cache_path))
            self.video_path = video_path
            self.frame_size = tuple(self.data['frame_size'])
            print(f"[PrecomputedTracker] Loaded {cache_path}")
            print(f"  - Frames: {self.data['total_frames']}")
            print(f"  - Grid: {self.data['grid_size']}")
            print(f"  - Size: {self.frame_size}")
            return True
        except Exception as e:
            print(f"[PrecomputedTracker] Failed to load: {e}")
            return False
    
    def initialize(self, frame: np.ndarray, annotations: List[Dict]) -> None:
        """Initialize with annotations (just store them, no processing needed)."""
        h, w = frame.shape[:2]
        self.frame_size = (h, w)
        self.current_frame = 0
        self.annotations = {}
        
        for ann in annotations:
            ann_id = ann['id']
            
            if 'points' in ann and ann['points'] and len(ann['points']) > 2:
                # Points are in percentage, convert to pixels
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
            
            self.annotations[ann_id] = {
                'initial_points': boundary_px.copy(),
                'current_points': boundary_px.copy(),
                'meta': ann
            }
        
        print(f"[PrecomputedTracker] Initialized {len(annotations)} annotations")
    
    def _interpolate_point(self, point: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, float]:
        """
        Interpolate a single point's position at given frame using bilinear interpolation
        from the 4 nearest grid points.
        
        Args:
            point: [x, y] in pixel coordinates at frame 0
            frame_idx: Target frame index
        
        Returns:
            (new_point, visibility)
        """
        if self.data is None:
            return point, 1.0
        
        coords = self.data['coords']  # [num_frames, num_points, 2]
        visibility = self.data['visibility']  # [num_frames, num_points]
        grid_size = self.data['grid_size']
        frame_h, frame_w = self.frame_size
        
        num_frames = coords.shape[0]
        frame_idx = min(frame_idx, num_frames - 1)
        
        # Get grid coordinates at frame 0
        coords_0 = coords[0]  # [N, 2] - all points at frame 0
        
        # Find 4 nearest grid points using grid structure
        gw, gh = grid_size[0], grid_size[1]
        
        # Grid points are evenly spaced
        cell_w = frame_w / gw
        cell_h = frame_h / gh
        
        # Find grid cell containing the point
        gx = point[0] / cell_w
        gy = point[1] / cell_h
        
        # Get surrounding grid indices
        gx0 = int(np.floor(gx))
        gy0 = int(np.floor(gy))
        gx1 = min(gx0 + 1, gw - 1)
        gy1 = min(gy0 + 1, gh - 1)
        gx0 = max(0, gx0)
        gy0 = max(0, gy0)
        
        # Bilinear weights
        wx = gx - gx0
        wy = gy - gy0
        
        # Grid point indices (row-major order)
        def grid_idx(x, y):
            return y * gw + x
        
        idx00 = grid_idx(gx0, gy0)
        idx10 = grid_idx(gx1, gy0)
        idx01 = grid_idx(gx0, gy1)
        idx11 = grid_idx(gx1, gy1)
        
        # Get displacements from frame 0 to frame_idx
        def get_displacement(idx):
            return coords[frame_idx, idx] - coords[0, idx]
        
        d00 = get_displacement(idx00)
        d10 = get_displacement(idx10)
        d01 = get_displacement(idx01)
        d11 = get_displacement(idx11)
        
        # Bilinear interpolation of displacement
        d_top = d00 * (1 - wx) + d10 * wx
        d_bottom = d01 * (1 - wx) + d11 * wx
        displacement = d_top * (1 - wy) + d_bottom * wy
        
        # Apply displacement
        new_point = point + displacement
        
        # Interpolate visibility
        v00 = visibility[frame_idx, idx00]
        v10 = visibility[frame_idx, idx10]
        v01 = visibility[frame_idx, idx01]
        v11 = visibility[frame_idx, idx11]
        v_top = v00 * (1 - wx) + v10 * wx
        v_bottom = v01 * (1 - wx) + v11 * wx
        vis = v_top * (1 - wy) + v_bottom * wy
        
        return new_point.astype(np.float32), float(vis)
    
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
            
            for pt in initial_pts:
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
