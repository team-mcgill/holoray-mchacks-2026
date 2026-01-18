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
import threading
from collections import OrderedDict

from tracker.core import DeformableTracker, TrackingResult
from tracker.affine_tracker import AffineInteriorTracker, AffineTrackingResult

# Try to import LiteTracker
try:
    from tracker.lite_tracker_wrapper import LiteTrackerWrapper, LiteTrackingResult
    LITE_TRACKER_AVAILABLE = True
    print("[TrackingService] LiteTracker available!")
except ImportError as e:
    LITE_TRACKER_AVAILABLE = False
    print(f"[TrackingService] LiteTracker not available: {e}")

# Import PrecomputedTracker for instant tracking from pre-processed data
import os
from tracker.precomputed_tracker import PrecomputedTracker, PrecomputedTrackingResult
PRECOMPUTED_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tracking_cache")

# Global cache for loaded precomputed tracking data (load once per video)
_precomputed_cache: Dict[str, dict] = {}

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
        # Store a deep copy of original annotations (before any tracking modifies them)
        import copy
        self.original_annotations = copy.deepcopy(annotations)
        self.annotations = annotations
        self.tracker_config = tracker_config or {}
        
        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.total_frames = 0
        self.frame_width = 0
        self.frame_height = 0
        
        # Check for pre-computed tracking data first (instant, best quality)
        self.precomputed_tracker = PrecomputedTracker(cache_dir=PRECOMPUTED_CACHE_DIR)
        self.use_precomputed = self.precomputed_tracker.has_precomputed_data(video_path)
        

        
        if self.use_precomputed:
            print("[Session] Using PrecomputedTracker (instant, pre-processed)")
            self.precomputed_tracker.load(video_path)
            self.tracker = self.precomputed_tracker
        elif LITE_TRACKER_AVAILABLE and self.tracker_config.get('use_lite_tracker', False):
            # LiteTracker is slow (~300ms/frame), only use if explicitly requested
            print("[Session] Using LiteTracker (accurate but slow)")
            self.tracker = LiteTrackerWrapper()
        else:
            print("[Session] Using Affine Interior tracker (real-time)")
            self.tracker = AffineInteriorTracker(
                num_interior_points=self.tracker_config.get('num_interior_points', 50)
            )
        
        # Keep reference for compatibility
        self.affine_tracker = self.tracker
        
        # State
        self.current_frame = 0
        self.is_running = False
        self.last_result: Optional[TrackingResult] = None
        
        # Predictive buffer
        self.buffer: OrderedDict[int, Dict] = OrderedDict()
        self.buffer_size = 120  # frames to track ahead
        self.buffer_lock = threading.Lock()
        self.tracking_thread: Optional[threading.Thread] = None
        self.stop_tracking = threading.Event()
        self.playback_frame = 0  # current frontend playback position
    
    def open_video(self, start_time: float = 0.0) -> bool:
        """Open video file and initialize tracker at given time."""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate start frame from time
        start_frame = int(start_time * self.fps)
        start_frame = max(0, min(start_frame, self.total_frames - 1))
        self.playback_frame = start_frame
        
        # Seek to start frame
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        self.affine_tracker.initialize(frame, self.annotations)
        self.current_frame = start_frame
        
        # Start predictive tracking thread
        self._start_predictive_tracking()
        
        return True

    def _start_predictive_tracking(self):
        """Start background thread for predictive tracking."""
        self.stop_tracking.clear()
        self.tracking_thread = threading.Thread(target=self._tracking_worker, daemon=True)
        self.tracking_thread.start()
        print(f"[Session] Started predictive tracking (buffer_size={self.buffer_size})")
    
    def _tracking_worker(self):
        """Background worker that tracks ahead of playback."""
        # If using precomputed, we can fill buffer instantly
        if self.use_precomputed:
            self._precomputed_buffer_fill()
            return
        
        # Create separate video capture for this thread
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        
        # Create separate tracker instance
        tracker = AffineInteriorTracker(
            num_interior_points=self.tracker_config.get('num_interior_points', 50)
        )
        
        # Initialize at frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return
        
        # Start from current playback position
        start_frame = self.playback_frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return
        
        # Store original annotations for initial frame (exact drawn position)
        # This ensures no coordinate transformation affects the first frame
        with self.buffer_lock:
            self.buffer[start_frame] = {
                'frame_idx': start_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'annotations': self.annotations,  # Use ORIGINAL annotations from frontend
                'processing_time_ms': 0,
                'method_used': 'initial',
                'current_time': start_frame / self.fps if self.fps > 0 else 0
            }
        
        tracker.initialize(frame, self.annotations)
        buffer_frame = start_frame
        
        while not self.stop_tracking.is_set():
            # Check if we need to track more frames
            with self.buffer_lock:
                frames_ahead = buffer_frame - self.playback_frame
                need_more = frames_ahead < self.buffer_size
            
            if not need_more:
                time.sleep(0.005)  # Wait a bit
                continue
            
            # Read and track next frame
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
                tracker.initialize(frame, self.annotations)
                buffer_frame = 0
                with self.buffer_lock:
                    self.buffer.clear()
                continue
            
            buffer_frame += 1
            
            # Track
            results = tracker.track_frame(frame)
            annotations_out = tracker.get_annotations_dict(results, frame.shape)
            
            result = {
                'frame_idx': buffer_frame,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'annotations': annotations_out,
                'processing_time_ms': 0,
                'method_used': 'predictive',
                'current_time': buffer_frame / self.fps if self.fps > 0 else 0
            }
            
            # Store in buffer
            with self.buffer_lock:
                self.buffer[buffer_frame] = result
                # Prune old frames (keep some behind playback for seeks)
                min_keep = max(0, self.playback_frame - 30)
                keys_to_remove = [k for k in self.buffer if k < min_keep]
                for k in keys_to_remove:
                    del self.buffer[k]
        
        cap.release()
    
    def _precomputed_buffer_fill(self):
        """Fill buffer using precomputed tracking data (instant)."""
        # Initialize tracker with annotations
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        
        # Initialize at playback_frame (where user drew annotation)
        # Use ORIGINAL annotations (not tracked/modified ones)
        self.precomputed_tracker.initialize(frame, self.original_annotations, start_frame=self.playback_frame)
        
        # Pre-fill buffer for all frames (instant lookups)
        while not self.stop_tracking.is_set():
            with self.buffer_lock:
                frames_ahead = len(self.buffer) - self.playback_frame
                if frames_ahead >= self.buffer_size:
                    time.sleep(0.01)
                    continue
                
                # Fill next batch of frames
                fill_start = max(self.buffer.keys()) + 1 if self.buffer else self.playback_frame
                
            # If we've reached the end, stop filling (video will loop, new session created)
            if fill_start >= self.total_frames:
                time.sleep(0.01)
                continue
            
            for frame_idx in range(fill_start, min(fill_start + 30, self.total_frames)):
                if self.stop_tracking.is_set():
                    break
                
                results = self.precomputed_tracker.get_annotations_for_frame(frame_idx)
                annotations_out = self.precomputed_tracker.get_annotations_dict(results, (self.frame_height, self.frame_width))
                
                result = {
                    'frame_idx': frame_idx,
                    'total_frames': self.total_frames,
                    'fps': self.fps,
                    'annotations': annotations_out,
                    'processing_time_ms': 0,
                    'method_used': 'precomputed',
                    'current_time': frame_idx / self.fps if self.fps > 0 else 0
                }
                
                with self.buffer_lock:
                    self.buffer[frame_idx] = result
                    # Prune old frames
                    min_keep = max(0, self.playback_frame - 30)
                    keys_to_remove = [k for k in self.buffer if k < min_keep]
                    for k in keys_to_remove:
                        del self.buffer[k]
    
    def get_annotations_for_frame(self, frame_idx: int) -> Optional[Dict]:
        """Get pre-computed annotations for a frame (O(1) lookup)."""
        # For precomputed tracker, compute directly - no buffer needed
        if self.use_precomputed:
            # Wrap around for looping videos
            wrapped_idx = frame_idx % self.total_frames if self.total_frames > 0 else frame_idx
            results = self.precomputed_tracker.get_annotations_for_frame(wrapped_idx)
            annotations_out = self.precomputed_tracker.get_annotations_dict(results, (self.frame_height, self.frame_width))
            return {
                'frame_idx': frame_idx,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'annotations': annotations_out,
                'processing_time_ms': 0,
                'method_used': 'precomputed',
                'current_time': frame_idx / self.fps if self.fps > 0 else 0
            }
        
        with self.buffer_lock:
            self.playback_frame = frame_idx
            return self.buffer.get(frame_idx)
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status for frontend."""
        with self.buffer_lock:
            frames = list(self.buffer.keys())
            return {
                'size': len(frames),
                'min_frame': min(frames) if frames else 0,
                'max_frame': max(frames) if frames else 0,
                'playback_frame': self.playback_frame,
                'frames_ahead': (max(frames) - self.playback_frame) if frames else 0,
                'ready': len(frames) >= 30  # Ready when we have 1 second buffered
            }
    
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
    
    def update_annotations(self, annotations: List[Dict], current_time: float = 0.0) -> None:
        """Update annotations being tracked - clears buffer and restarts tracking."""
        import copy
        self.original_annotations = copy.deepcopy(annotations)  # Store clean copy
        self.annotations = annotations
        self.playback_frame = int(current_time * self.fps)  # Start from correct frame
        
        # Stop current tracking thread
        self.stop_tracking.set()
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
        
        # Clear buffer
        with self.buffer_lock:
            self.buffer.clear()
        
        # Re-initialize main tracker
        if self.cap is not None:
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
            ret, frame = self.cap.read()
            if ret:
                self.affine_tracker.initialize(frame, annotations)
        
        # Restart predictive tracking from current playback position
        self._start_predictive_tracking()
        print(f"[Session] Annotations updated, buffer cleared, tracking restarted")
    
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
        """Release video capture and stop tracking thread."""
        self.stop_tracking.set()
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=1.0)
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
        annotations: List[Dict],
        start_time: float = 0.0
    ) -> bool:
        """Create a new tracking session."""
        if session_id in self.sessions:
            self.sessions[session_id].close()
        
        session = VideoTrackingSession(video_path, annotations)
        if not session.open_video(start_time):
            return False
        
        self.sessions[session_id] = session
        return True
    
    def get_session(self, session_id: str) -> Optional[VideoTrackingSession]:
        """Get existing session."""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str) -> None:
        """Close and remove session."""
        session = self.sessions.pop(session_id, None)
        if session:
            session.close()
    
    def close_all(self) -> None:
        """Close all sessions."""
        for session in self.sessions.values():
            session.close()
        self.sessions.clear()


# Global manager instance
tracking_manager = TrackingManager()
