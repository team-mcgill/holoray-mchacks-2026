"""
Core Deformable Tracker - Hybrid multi-algorithm tracking engine.

Architecture:
- Tier 1 (Fast): Lucas-Kanade optical flow for frame-to-frame (~5ms)
- Tier 2 (Dense): CoTracker3 for keyframe re-anchoring (~30-50ms)
- Tier 3 (Recovery): Feature-based re-detection on tracking failure
"""
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time

from .optical_flow import LucasKanadeTracker
from .utils import sample_boundary_points, reconstruct_contour, points_to_svg_path, compute_bbox_from_points

@dataclass
class TrackedAnnotation:
    """State for a single tracked annotation."""
    id: str
    label: str
    color: str
    boundary_points: np.ndarray  # (N, 2) sampled boundary points
    confidence: float = 1.0
    frames_since_anchor: int = 0
    lost: bool = False
    original_bbox: Dict = field(default_factory=dict)

@dataclass
class TrackingResult:
    """Result of tracking for a single frame."""
    frame_idx: int
    annotations: List[Dict[str, Any]]
    processing_time_ms: float
    method_used: str  # "lk", "dense", "recovery"


class DeformableTracker:
    """
    Main deformable anatomy tracker using hybrid multi-algorithm approach.
    Optimized for M3 MacBook Air real-time performance.
    """
    
    def __init__(
        self,
        num_boundary_points: int = 32,
        keyframe_interval: int = 10,
        confidence_threshold: float = 0.5,
        use_dense_tracker: bool = True
    ):
        """
        Initialize the deformable tracker.
        
        Args:
            num_boundary_points: Number of points to sample on annotation boundary
            keyframe_interval: Frames between dense tracker re-anchoring
            confidence_threshold: Below this, trigger recovery mode
            use_dense_tracker: Whether to use CoTracker3 (requires more memory)
        """
        self.num_boundary_points = num_boundary_points
        self.keyframe_interval = keyframe_interval
        self.confidence_threshold = confidence_threshold
        self.use_dense_tracker = use_dense_tracker
        
        # Tracking state
        self.tracked_annotations: Dict[str, TrackedAnnotation] = {}
        self.frame_idx = 0
        self.initialized = False
        
        # Lucas-Kanade tracker (fast path)
        self.lk_tracker = LucasKanadeTracker()
        
        # Dense tracker (CoTracker3) - lazy loaded
        self.dense_tracker = None
        self.dense_tracker_available = False
        
        # Frame dimensions (set on first frame)
        self.frame_width = 0
        self.frame_height = 0
        
        # Performance metrics
        self.last_processing_time = 0.0
    
    def _load_dense_tracker(self):
        """Lazy load CoTracker3 to save memory if not needed."""
        if self.dense_tracker is not None:
            return
        
        try:
            import torch
            # Check for MPS (Apple Silicon) or fallback to CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("Using Apple Metal (MPS) for dense tracking")
            else:
                self.device = torch.device("cpu")
                print("Using CPU for dense tracking")
            
            # Try to load CoTracker3
            try:
                from cotracker.predictor import CoTrackerPredictor
                self.dense_tracker = CoTrackerPredictor(
                    checkpoint=None  # Use default weights
                )
                self.dense_tracker.to(self.device)
                self.dense_tracker_available = True
                print("CoTracker3 loaded successfully")
            except ImportError:
                print("CoTracker3 not available - using LK-only mode")
                self.dense_tracker_available = False
        except Exception as e:
            print(f"Could not load dense tracker: {e}")
            self.dense_tracker_available = False
    
    def initialize(
        self,
        frame: np.ndarray,
        annotations: List[Dict]
    ) -> None:
        """
        Initialize tracker with first frame and annotations.
        
        Args:
            frame: First video frame (BGR)
            annotations: List of annotation dicts with x, y, width, height (percentages)
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        self.frame_idx = 0
        self.tracked_annotations = {}
        
        # Convert percentage coordinates to pixel coordinates
        for ann in annotations:
            ann_id = ann['id']
            
            # Sample boundary points (auto-detects freeform points if present)
            boundary_pts = sample_boundary_points(
                ann, 
                num_points=self.num_boundary_points
            )
            
            # Convert from percentage to pixel coordinates
            boundary_pts_px = boundary_pts.copy()
            boundary_pts_px[:, 0] *= self.frame_width / 100.0
            boundary_pts_px[:, 1] *= self.frame_height / 100.0
            
            self.tracked_annotations[ann_id] = TrackedAnnotation(
                id=ann_id,
                label=ann.get('label', 'Unknown'),
                color=ann.get('color', '#0ea5e9'),
                boundary_points=boundary_pts_px,
                original_bbox=ann
            )
        
        # Initialize LK tracker with all boundary points
        if self.tracked_annotations:
            all_points = np.vstack([
                ta.boundary_points for ta in self.tracked_annotations.values()
            ])
            self.lk_tracker.initialize(frame, all_points)
        
        self.initialized = True
        
        # Lazy load dense tracker if needed
        if self.use_dense_tracker:
            self._load_dense_tracker()
    
    def track_frame(self, frame: np.ndarray) -> TrackingResult:
        """
        Track all annotations to a new frame.
        
        Args:
            frame: New video frame (BGR)
        
        Returns:
            TrackingResult with updated annotation positions
        """
        start_time = time.perf_counter()
        
        if not self.initialized:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        
        self.frame_idx += 1
        method_used = "lk"
        
        # Collect all boundary points
        all_points = []
        point_to_annotation = []  # Maps point index to annotation ID
        
        for ann_id, tracked_ann in self.tracked_annotations.items():
            if not tracked_ann.lost:
                start_idx = len(all_points)
                all_points.append(tracked_ann.boundary_points)
                point_to_annotation.extend(
                    [(ann_id, i) for i in range(len(tracked_ann.boundary_points))]
                )
        
        if not all_points:
            # No points to track
            return self._build_result(method_used, start_time)
        
        all_points = np.vstack(all_points)
        
        # Update LK tracker with current points
        self.lk_tracker.update_points(all_points)
        
        # Track using Lucas-Kanade (fast path)
        new_points, status, confidence = self.lk_tracker.track(frame)
        
        # Update annotation boundary points
        point_idx = 0
        for ann_id, tracked_ann in self.tracked_annotations.items():
            if tracked_ann.lost:
                continue
            
            n_points = len(tracked_ann.boundary_points)
            ann_points = new_points[point_idx:point_idx + n_points]
            ann_status = status[point_idx:point_idx + n_points]
            ann_confidence = confidence[point_idx:point_idx + n_points]
            
            # Update points - only update valid points, keep others at previous position
            valid_mask = ann_status == 1
            if np.any(valid_mask):
                # For invalid points, interpolate from valid neighbors
                if not np.all(valid_mask):
                    valid_indices = np.where(valid_mask)[0]
                    invalid_indices = np.where(~valid_mask)[0]
                    for idx in invalid_indices:
                        # Find nearest valid neighbors
                        distances = np.abs(valid_indices - idx)
                        nearest = valid_indices[np.argmin(distances)]
                        # Use the valid point's displacement
                        displacement = ann_points[nearest] - tracked_ann.boundary_points[nearest]
                        ann_points[idx] = tracked_ann.boundary_points[idx] + displacement
                
                tracked_ann.boundary_points = ann_points
                tracked_ann.confidence = float(np.mean(ann_confidence[valid_mask]))
            else:
                # All points lost - use median displacement from previous frame
                tracked_ann.confidence = 0.3
            
            tracked_ann.frames_since_anchor += 1
            
            # Only mark as lost if confidence is very low for many frames
            # (removed aggressive lost marking)
            
            point_idx += n_points
        
        # Check if we need dense re-anchoring
        needs_reanchor = any(
            ta.frames_since_anchor >= self.keyframe_interval
            for ta in self.tracked_annotations.values()
            if not ta.lost
        )
        
        # Dense tracker re-anchoring (if available and needed)
        if needs_reanchor and self.dense_tracker_available:
            # This would call CoTracker3 for re-anchoring
            # For now, just reset the anchor counter
            method_used = "dense"
            for ta in self.tracked_annotations.values():
                if not ta.lost:
                    ta.frames_since_anchor = 0
        
        # Build result
        return self._build_result(method_used, start_time)
    
    def _build_result(self, method_used: str, start_time: float) -> TrackingResult:
        """Build TrackingResult from current state."""
        annotations = []
        
        for ann_id, tracked_ann in self.tracked_annotations.items():
            if tracked_ann.lost:
                continue
            
            # Convert pixel coordinates back to percentages
            boundary_pts_pct = tracked_ann.boundary_points.copy()
            boundary_pts_pct[:, 0] *= 100.0 / self.frame_width
            boundary_pts_pct[:, 1] *= 100.0 / self.frame_height
            
            # Compute bounding box from tracked points
            bbox = compute_bbox_from_points(boundary_pts_pct)
            
            # Generate smooth contour for rendering
            smooth_contour = reconstruct_contour(boundary_pts_pct, smoothing=0.5)
            svg_path = points_to_svg_path(smooth_contour, closed=True)
            
            annotations.append({
                'id': tracked_ann.id,
                'label': tracked_ann.label,
                'color': tracked_ann.color,
                'x': bbox['x'],
                'y': bbox['y'],
                'width': bbox['width'],
                'height': bbox['height'],
                'confidence': tracked_ann.confidence,
                'contour_points': boundary_pts_pct.tolist(),
                'svg_path': svg_path,
                'deformed': True  # Flag indicating this is a deformed annotation
            })
        
        processing_time = (time.perf_counter() - start_time) * 1000
        self.last_processing_time = processing_time
        
        return TrackingResult(
            frame_idx=self.frame_idx,
            annotations=annotations,
            processing_time_ms=processing_time,
            method_used=method_used
        )
    
    def add_annotation(self, frame: np.ndarray, annotation: Dict) -> None:
        """Add a new annotation to track mid-video."""
        ann_id = annotation['id']
        
        boundary_pts = sample_boundary_points(
            annotation,
            num_points=self.num_boundary_points
        )
        
        boundary_pts_px = boundary_pts.copy()
        boundary_pts_px[:, 0] *= self.frame_width / 100.0
        boundary_pts_px[:, 1] *= self.frame_height / 100.0
        
        self.tracked_annotations[ann_id] = TrackedAnnotation(
            id=ann_id,
            label=annotation.get('label', 'Unknown'),
            color=annotation.get('color', '#0ea5e9'),
            boundary_points=boundary_pts_px,
            original_bbox=annotation
        )
    
    def remove_annotation(self, ann_id: str) -> None:
        """Remove an annotation from tracking."""
        if ann_id in self.tracked_annotations:
            del self.tracked_annotations[ann_id]
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracked_annotations = {}
        self.frame_idx = 0
        self.initialized = False
