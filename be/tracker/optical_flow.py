"""
Lucas-Kanade Optical Flow tracker for fast frame-to-frame tracking.
This is the "fast path" that handles small inter-frame motions efficiently.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List

class LucasKanadeTracker:
    """
    Fast optical flow tracker using Lucas-Kanade method.
    Optimized for Apple Silicon via OpenCV's NEON acceleration.
    """
    
    def __init__(
        self,
        win_size: Tuple[int, int] = (21, 21),
        max_level: int = 3,
        criteria: Optional[Tuple] = None
    ):
        """
        Initialize Lucas-Kanade tracker.
        
        Args:
            win_size: Window size for optical flow
            max_level: Maximum pyramid level
            criteria: Termination criteria (iterations, epsilon)
        """
        self.win_size = win_size
        self.max_level = max_level
        
        if criteria is None:
            self.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,  # max iterations
                0.01  # epsilon
            )
        else:
            self.criteria = criteria
        
        self.lk_params = dict(
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria
        )
        
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
    
    def initialize(self, frame: np.ndarray, points: np.ndarray) -> None:
        """
        Initialize tracker with first frame and points.
        
        Args:
            frame: BGR or grayscale frame
            points: np.ndarray of shape (N, 2) with point coordinates
        """
        if len(frame.shape) == 3:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_gray = frame.copy()
        
        # Ensure points are float32 and properly shaped for OpenCV
        self.prev_points = points.astype(np.float32).reshape(-1, 1, 2)
    
    def track(
        self, 
        frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track points to new frame.
        
        Args:
            frame: New BGR or grayscale frame
        
        Returns:
            Tuple of:
                - new_points: np.ndarray of shape (N, 2)
                - status: np.ndarray of shape (N,) with 1 for tracked, 0 for lost
                - confidence: np.ndarray of shape (N,) with tracking quality
        """
        if self.prev_gray is None or self.prev_points is None:
            raise RuntimeError("Tracker not initialized. Call initialize() first.")
        
        if len(frame.shape) == 3:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = frame
        
        # Forward tracking
        new_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_points,
            None,
            **self.lk_params
        )
        
        # Backward tracking for validation (forward-backward consistency check)
        if new_points is not None:
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                curr_gray,
                self.prev_gray,
                new_points,
                None,
                **self.lk_params
            )
            
            # Compute forward-backward error
            fb_error = np.linalg.norm(
                self.prev_points.reshape(-1, 2) - back_points.reshape(-1, 2),
                axis=1
            )
            
            # Points with high FB error are unreliable
            fb_threshold = 2.0  # pixels (increased for medical video)
            fb_valid = fb_error < fb_threshold
            
            status = status.flatten() & back_status.flatten() & fb_valid.astype(np.uint8)
            
            # Apply shape regularization - prevent points from drifting too far from neighbors
            new_points_2d = new_points.reshape(-1, 2)
            prev_points_2d = self.prev_points.reshape(-1, 2)
            
            # Compute displacement of each point
            displacements = new_points_2d - prev_points_2d
            
            # Compute median displacement (robust to outliers)
            median_disp = np.median(displacements, axis=0)
            
            # For each point, check if its displacement is an outlier
            disp_from_median = np.linalg.norm(displacements - median_disp, axis=1)
            median_dev = np.median(disp_from_median)
            
            # Points that move very differently from the group are suspicious
            outlier_threshold = max(3.0, median_dev * 3)  # 3 MAD or 3 pixels minimum
            outliers = disp_from_median > outlier_threshold
            
            # For outlier points, use the median displacement instead
            for i in np.where(outliers)[0]:
                new_points_2d[i] = prev_points_2d[i] + median_disp
            
            new_points = new_points_2d.reshape(-1, 1, 2)
        
        # Compute confidence based on tracking error
        if err is not None:
            # Normalize error to 0-1 range (lower error = higher confidence)
            confidence = 1.0 - np.clip(err.flatten() / 50.0, 0, 1)
        else:
            confidence = np.ones(len(self.prev_points))
        
        # Update state
        self.prev_gray = curr_gray
        if new_points is not None:
            self.prev_points = new_points
        
        # Reshape output
        if new_points is not None:
            output_points = new_points.reshape(-1, 2)
        else:
            output_points = self.prev_points.reshape(-1, 2)
        
        return output_points, status.flatten(), confidence
    
    def update_points(self, points: np.ndarray) -> None:
        """
        Update tracked points (e.g., after re-anchoring from dense tracker).
        
        Args:
            points: New point coordinates
        """
        self.prev_points = points.astype(np.float32).reshape(-1, 1, 2)
    
    def get_average_confidence(self) -> float:
        """Get average tracking confidence across all points."""
        # This is a placeholder - actual confidence would come from last track() call
        return 0.8


class MultiScaleLKTracker(LucasKanadeTracker):
    """
    Multi-scale Lucas-Kanade tracker for handling both small and large motions.
    """
    
    def __init__(self):
        super().__init__(
            win_size=(31, 31),  # Larger window for robustness
            max_level=4  # More pyramid levels for large motion
        )
    
    def track_with_scale_adaptation(
        self,
        frame: np.ndarray,
        expected_motion_scale: str = "auto"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track with adaptive parameters based on expected motion scale.
        
        Args:
            frame: New frame
            expected_motion_scale: "small", "medium", "large", or "auto"
        
        Returns:
            Same as track()
        """
        # Adjust parameters based on expected motion
        if expected_motion_scale == "small":
            self.lk_params['winSize'] = (15, 15)
            self.lk_params['maxLevel'] = 2
        elif expected_motion_scale == "large":
            self.lk_params['winSize'] = (41, 41)
            self.lk_params['maxLevel'] = 5
        else:
            self.lk_params['winSize'] = (21, 21)
            self.lk_params['maxLevel'] = 3
        
        return self.track(frame)
