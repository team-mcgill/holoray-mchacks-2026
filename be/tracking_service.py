"""
Real-time tracking service with WebSocket support.
Handles video processing and streams tracking results to frontend.
"""
import asyncio
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import time

from tracker.core import DeformableTracker, TrackingResult
from tracker.affine_tracker import AffineInteriorTracker, AffineTrackingResult

class VideoTrackingSession:
    """
    Manages a tracking session for a single video.
    """
    
    def __init__(
        self,
        video_path: str,
        annotations: List[Dict],
        tracker_config: Optional[Dict] = None
    ):
        """
        Initialize tracking session.
        
        Args:
            video_path: Path to video file
            annotations: Initial annotations to track
            tracker_config: Optional tracker configuration
        """
        self.video_path = video_path
        self.annotations = annotations
        self.tracker_config = tracker_config or {}
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Use Affine Interior Tracker (fast + no drift)
        print("[Session] Using Affine Interior tracker")
        self.affine_tracker = AffineInteriorTracker(
            num_interior_points=self.tracker_config.get('num_interior_points', 50)
        )
        
        # State
        self.current_frame = 0
        self.is_running = False
        self.last_result: Optional[TrackingResult] = None
    
    def open_video(self) -> bool:
        """Open video file and initialize tracker."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame and initialize tracker
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        self.affine_tracker.initialize(frame, self.annotations)
        self.current_frame = 0
        
        return True
    
    def track_next_frame(self) -> Optional[Dict]:
        """
        Track annotations to next frame.
        
        Returns:
            Dict with tracking results or None if video ended
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            # Video ended or loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None
            # Re-initialize tracker at video loop
            self.affine_tracker.initialize(frame, self.annotations)
            self.current_frame = 0
            return {
                'frame_idx': 0,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'annotations': self.annotations,
                'processing_time_ms': 0,
                'method_used': 'affine_loop',
                'current_time': 0
            }
        
        self.current_frame += 1
        
        import time
        start = time.perf_counter()
        
        results = self.affine_tracker.track_frame(frame)
        annotations_out = self.affine_tracker.get_annotations_dict(results, frame.shape)
        
        processing_time = (time.perf_counter() - start) * 1000
        
        return {
            'frame_idx': self.current_frame,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'annotations': annotations_out,
            'processing_time_ms': processing_time,
            'method_used': 'affine',
            'current_time': self.current_frame / self.fps if self.fps > 0 else 0
        }
    
    def seek_to_frame(self, frame_idx: int) -> Optional[Dict]:
        """Seek to specific frame and re-initialize tracking."""
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        self.affine_tracker.initialize(frame, self.annotations)
        self.current_frame = frame_idx
        
        return {
            'frame_idx': frame_idx,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'annotations': self.annotations,
            'processing_time_ms': 0,
            'method_used': 'affine_seek',
            'current_time': frame_idx / self.fps if self.fps > 0 else 0
        }
    
    def update_annotations(self, annotations: List[Dict]) -> None:
        """Update annotations being tracked."""
        self.annotations = annotations
        
        # Re-initialize tracker with new annotations on current frame
        if self.cap is not None:
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            # Go back one frame since we already read it
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
            ret, frame = self.cap.read()
            if ret:
                self.affine_tracker.initialize(frame, annotations)
    
    def track_at_time(self, time_seconds: float) -> Optional[Dict]:
        """
        Track annotations at a specific video time.
        This syncs the tracker with the frontend video playback.
        """
        if self.cap is None:
            return None
        
        # Calculate frame index from time
        target_frame = int(time_seconds * self.fps)
        target_frame = max(0, min(target_frame, self.total_frames - 1))
        
        frame_diff = target_frame - self.current_frame
        
        # Going backward or very large jump - need to seek and re-init
        if frame_diff < 0 or frame_diff > 30:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            self.affine_tracker.initialize(frame, self.annotations)
            self.current_frame = target_frame
            # Return annotations with points from the affine tracker
            init_annotations = self.affine_tracker.get_annotations_dict(
                {ann_id: AffineTrackingResult(
                    boundary_points=data['boundary_px'],
                    interior_points=data['interior_px'],
                    transform=np.eye(2, 3, dtype=np.float32),
                    confidence=1.0,
                    num_inliers=len(data['interior_px'])
                ) for ann_id, data in self.affine_tracker.annotations.items()},
                frame.shape
            )
            return {
                'frame_idx': target_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'annotations': init_annotations,
                'processing_time_ms': 0,
                'method_used': 'affine_seek',
                'current_time': target_frame / self.fps if self.fps > 0 else 0
            }
        
        # Track forward through intermediate frames
        result = None
        while self.current_frame < target_frame:
            result = self.track_next_frame()
            if result is None:
                break
        
        if result:
            return result
        elif self.last_result:
            return self._result_to_dict(self.last_result)
        return None
    
    def _result_to_dict(self, result: TrackingResult) -> Dict:
        """Convert TrackingResult to dictionary for JSON serialization."""
        return {
            'frame_idx': result.frame_idx,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'annotations': result.annotations,
            'processing_time_ms': result.processing_time_ms,
            'method_used': result.method_used,
            'current_time': result.frame_idx / self.fps if self.fps > 0 else 0
        }
    
    def close(self) -> None:
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class TrackingManager:
    """
    Manages multiple tracking sessions.
    """
    
    def __init__(self):
        self.sessions: Dict[str, VideoTrackingSession] = {}
    
    def create_session(
        self,
        session_id: str,
        video_path: str,
        annotations: List[Dict]
    ) -> bool:
        """Create a new tracking session."""
        if session_id in self.sessions:
            self.sessions[session_id].close()
        
        session = VideoTrackingSession(video_path, annotations)
        if not session.open_video():
            return False
        
        self.sessions[session_id] = session
        return True
    
    def get_session(self, session_id: str) -> Optional[VideoTrackingSession]:
        """Get existing session."""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str) -> None:
        """Close and remove session."""
        if session_id in self.sessions:
            self.sessions[session_id].close()
            del self.sessions[session_id]
    
    def close_all(self) -> None:
        """Close all sessions."""
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()


# Global manager instance
tracking_manager = TrackingManager()
