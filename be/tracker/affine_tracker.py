"""
Affine Interior Tracker - tracks interior points and applies affine transform to boundary.
Solves the boundary drift problem by not tracking boundary points directly.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class AffineTrackingResult:
    """Result from affine tracking."""
    boundary_points: np.ndarray  # Transformed boundary points
    interior_points: np.ndarray  # Tracked interior points
    transform: np.ndarray  # 2x3 affine matrix
    confidence: float
    num_inliers: int


class AffineInteriorTracker:
    """
    Tracks objects using CSRT tracker (best for microscopy/textured regions).
    Falls back to LK optical flow if needed.
    """
    
    def __init__(
        self,
        num_interior_points: int = 50,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        ransac_threshold: float = 3.0,
        use_csrt: bool = True  # Use CSRT tracker instead of LK
    ):
        self.num_interior_points = num_interior_points
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.ransac_threshold = ransac_threshold
        self.use_csrt = use_csrt
        
        self.lk_params = dict(
            winSize=lk_win_size,
            maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # State per annotation
        self.annotations: Dict[str, dict] = {}
        self.prev_gray: Optional[np.ndarray] = None
        self.csrt_trackers: Dict[str, cv2.Tracker] = {}  # CSRT trackers per annotation
        
        # Downscaling for performance (CSRT is slow on large frames)
        self.max_tracking_dim = 640
        self.tracking_scale = 1.0
    
    def _sample_interior_points(
        self,
        boundary_points: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Sample points inside the polygon boundary."""
        h, w = frame_shape[:2]
        
        # Create mask from boundary
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = boundary_points.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        
        # Find good features inside the mask
        interior = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.num_interior_points,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask
        )
        
        if interior is None or len(interior) < 4:
            # Fallback: sample random points inside polygon
            bbox = cv2.boundingRect(pts)
            x, y, bw, bh = bbox
            
            points = []
            for _ in range(self.num_interior_points * 10):
                px = np.random.randint(x, x + bw)
                py = np.random.randint(y, y + bh)
                if mask[py, px] > 0:
                    points.append([px, py])
                if len(points) >= self.num_interior_points:
                    break
            
            if len(points) < 4:
                # Last resort: use centroid and scaled boundary
                centroid = boundary_points.mean(axis=0)
                scaled = centroid + (boundary_points - centroid) * 0.5
                return scaled.astype(np.float32)
            
            return np.array(points, dtype=np.float32)
        
        return interior.reshape(-1, 2)
    
    def initialize(
        self,
        frame: np.ndarray,
        annotations: List[Dict]
    ) -> None:
        """
        Initialize tracker with first frame and annotations.
        
        Args:
            frame: First video frame (BGR)
            annotations: List of annotation dicts with 'id' and 'points' (boundary)
        """
        if len(frame.shape) == 3:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_gray = frame.copy()
        
        h, w = frame.shape[:2]
        
        # Calculate downscale factor for tracking
        self.tracking_scale = min(1.0, self.max_tracking_dim / max(h, w))
        
        self.annotations = {}
        
        for ann in annotations:
            ann_id = ann['id']
            
            # Get boundary points in pixel coordinates
            if 'points' in ann and ann['points'] and len(ann['points']) > 2:
                boundary_pct = np.array(ann['points'], dtype=np.float32)
                boundary_px = boundary_pct.copy()
                boundary_px[:, 0] = boundary_px[:, 0] * w / 100.0
                boundary_px[:, 1] = boundary_px[:, 1] * h / 100.0
            else:
                # Create boundary from bounding box
                x = ann['x'] * w / 100.0
                y = ann['y'] * h / 100.0
                bw = ann['width'] * w / 100.0
                bh = ann['height'] * h / 100.0
                boundary_px = np.array([
                    [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
                ], dtype=np.float32)
            
            # Sample interior points
            interior_px = self._sample_interior_points(boundary_px, frame.shape)
            
            self.annotations[ann_id] = {
                'boundary_px': boundary_px,
                'interior_px': interior_px,
                'original_boundary_px': boundary_px.copy(),
                'meta': ann
            }
            
            # Initialize CSRT tracker for this annotation (on scaled frame)
            if self.use_csrt:
                x_min, y_min = boundary_px.min(axis=0)
                x_max, y_max = boundary_px.max(axis=0)
                
                # Scale bbox for tracking
                s = self.tracking_scale
                scaled_bbox = (int(x_min * s), int(y_min * s), 
                              int((x_max - x_min) * s), int((y_max - y_min) * s))
                
                # Create scaled frame for tracker init
                if s < 1.0:
                    scaled_frame = cv2.resize(frame, None, fx=s, fy=s)
                else:
                    scaled_frame = frame
                
                tracker = cv2.TrackerCSRT_create()
                tracker.init(scaled_frame, scaled_bbox)
                self.csrt_trackers[ann_id] = tracker
        
        mode = "CSRT" if self.use_csrt else "LK optical flow"
        print(f"[Tracker] Initialized {len(annotations)} annotations with {mode}")
    
    def _track_with_csrt(self, frame: np.ndarray) -> Dict[str, AffineTrackingResult]:
        """Track using CSRT trackers."""
        results = {}
        
        # Downscale frame for faster tracking
        s = self.tracking_scale
        if s < 1.0:
            scaled_frame = cv2.resize(frame, None, fx=s, fy=s)
        else:
            scaled_frame = frame
        
        for ann_id, data in self.annotations.items():
            prev_boundary = data['boundary_px']
            tracker = self.csrt_trackers.get(ann_id)
            
            if tracker is None:
                results[ann_id] = AffineTrackingResult(
                    boundary_points=prev_boundary,
                    interior_points=data['interior_px'],
                    transform=np.eye(2, 3, dtype=np.float32),
                    confidence=0.0,
                    num_inliers=0
                )
                continue
            
            success, bbox = tracker.update(scaled_frame)
            
            if success:
                x, y, w, h = bbox
                
                # Scale bbox back to original resolution
                if s < 1.0:
                    x, y, w, h = x / s, y / s, w / s, h / s
                
                # Compute how the bounding box changed
                old_x_min, old_y_min = prev_boundary.min(axis=0)
                old_x_max, old_y_max = prev_boundary.max(axis=0)
                old_w = old_x_max - old_x_min
                old_h = old_y_max - old_y_min
                old_cx = (old_x_min + old_x_max) / 2
                old_cy = (old_y_min + old_y_max) / 2
                
                new_cx = x + w / 2
                new_cy = y + h / 2
                scale_x = w / old_w if old_w > 0 else 1
                scale_y = h / old_h if old_h > 0 else 1
                
                # Transform boundary: center, scale, translate
                new_boundary = prev_boundary - np.array([old_cx, old_cy])
                new_boundary[:, 0] *= scale_x
                new_boundary[:, 1] *= scale_y
                new_boundary += np.array([new_cx, new_cy])
                
                confidence = 1.0
            else:
                new_boundary = prev_boundary
                confidence = 0.0
            
            results[ann_id] = AffineTrackingResult(
                boundary_points=new_boundary.astype(np.float32),
                interior_points=data['interior_px'],
                transform=np.eye(2, 3, dtype=np.float32),
                confidence=confidence,
                num_inliers=100 if success else 0
            )
            
            data['boundary_px'] = new_boundary.astype(np.float32)
        
        return results
    
    def track_frame(self, frame: np.ndarray) -> Dict[str, AffineTrackingResult]:
        """
        Track all annotations to new frame.
        
        Args:
            frame: New video frame (BGR)
        
        Returns:
            Dict mapping annotation ID to AffineTrackingResult
        """
        # Use CSRT tracking if enabled
        if self.use_csrt:
            return self._track_with_csrt(frame)
        
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame.copy()
        
        results = {}
        
        for ann_id, data in self.annotations.items():
            prev_interior = data['interior_px'].reshape(-1, 1, 2).astype(np.float32)
            prev_boundary = data['boundary_px']
            
            # Track interior points with LK
            new_interior, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                curr_gray,
                prev_interior,
                None,
                **self.lk_params
            )
            
            if new_interior is None:
                results[ann_id] = AffineTrackingResult(
                    boundary_points=prev_boundary,
                    interior_points=prev_interior.reshape(-1, 2),
                    transform=np.eye(2, 3, dtype=np.float32),
                    confidence=0.0,
                    num_inliers=0
                )
                continue
            
            new_interior = new_interior.reshape(-1, 2)
            status = status.flatten()
            
            # Filter to successfully tracked points
            valid_mask = status == 1
            if valid_mask.sum() < 3:
                # Not enough points, use translation only
                if valid_mask.sum() > 0:
                    disp = (new_interior[valid_mask] - prev_interior.reshape(-1, 2)[valid_mask]).mean(axis=0)
                    new_boundary = prev_boundary + disp
                else:
                    new_boundary = prev_boundary
                
                results[ann_id] = AffineTrackingResult(
                    boundary_points=new_boundary,
                    interior_points=new_interior,
                    transform=np.eye(2, 3, dtype=np.float32),
                    confidence=0.3,
                    num_inliers=int(valid_mask.sum())
                )
                data['boundary_px'] = new_boundary
                data['interior_px'] = new_interior
                continue
            
            src_pts = prev_interior.reshape(-1, 2)[valid_mask]
            dst_pts = new_interior[valid_mask]
            
            # Estimate affine transformation with RANSAC
            transform, inliers = cv2.estimateAffinePartial2D(
                src_pts.reshape(-1, 1, 2),
                dst_pts.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold
            )
            
            if transform is None:
                # Fallback to simple translation
                disp = (dst_pts - src_pts).mean(axis=0)
                new_boundary = prev_boundary + disp
                transform = np.array([[1, 0, disp[0]], [0, 1, disp[1]]], dtype=np.float32)
                num_inliers = len(src_pts)
            else:
                # Apply affine transform to boundary
                ones = np.ones((len(prev_boundary), 1))
                boundary_h = np.hstack([prev_boundary, ones])
                new_boundary = (transform @ boundary_h.T).T
                num_inliers = int(inliers.sum()) if inliers is not None else len(src_pts)
            
            confidence = num_inliers / len(src_pts) if len(src_pts) > 0 else 0
            
            results[ann_id] = AffineTrackingResult(
                boundary_points=new_boundary.astype(np.float32),
                interior_points=new_interior,
                transform=transform,
                confidence=confidence,
                num_inliers=num_inliers
            )
            
            # Update state
            data['boundary_px'] = new_boundary.astype(np.float32)
            data['interior_px'] = new_interior
        
        self.prev_gray = curr_gray
        return results
    
    def get_annotations_dict(
        self,
        results: Dict[str, AffineTrackingResult],
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
                'num_inliers': result.num_inliers
            })
        
        return output
