"""
Test script for the deformable tracker.
Run this to verify tracking works on your M3 MacBook Air.
"""
import cv2
import numpy as np
import time
from tracker.core import DeformableTracker
from tracker.utils import sample_boundary_points

def test_tracker_on_video(video_path: str, num_frames: int = 100):
    """
    Test tracker performance on a video file.
    """
    print(f"\n{'='*60}")
    print(f"Testing Deformable Tracker on: {video_path}")
    print(f"{'='*60}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame")
        return
    
    # Create test annotations (simulating user-drawn boxes)
    # Place one in the center-ish area where anatomy would be
    test_annotations = [
        {
            'id': 'test_annotation_1',
            'label': 'Left Ventricle',
            'color': '#0ea5e9',
            'x': 30.0,  # percentage
            'y': 30.0,
            'width': 40.0,
            'height': 40.0
        }
    ]
    
    # Initialize tracker
    tracker = DeformableTracker(
        num_boundary_points=32,
        keyframe_interval=10,
        use_dense_tracker=False  # LK-only mode for speed test
    )
    
    print(f"\nInitializing tracker with {len(test_annotations)} annotation(s)...")
    tracker.initialize(frame, test_annotations)
    print("Tracker initialized!")
    
    # Track through frames and measure performance
    processing_times = []
    frames_processed = 0
    
    print(f"\nTracking {min(num_frames, total_frames)} frames...")
    print("-" * 40)
    
    start_total = time.perf_counter()
    
    while frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
            tracker.initialize(frame, test_annotations)
            continue
        
        # Track
        result = tracker.track_frame(frame)
        processing_times.append(result.processing_time_ms)
        frames_processed += 1
        
        # Print progress every 20 frames
        if frames_processed % 20 == 0:
            avg_time = np.mean(processing_times[-20:])
            print(f"  Frame {frames_processed}: {avg_time:.2f}ms avg ({1000/avg_time:.1f} FPS)")
    
    total_time = time.perf_counter() - start_total
    
    # Print results
    print(f"\n{'='*60}")
    print("PERFORMANCE RESULTS")
    print(f"{'='*60}")
    print(f"Frames processed: {frames_processed}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {frames_processed/total_time:.1f}")
    print(f"\nPer-frame processing time:")
    print(f"  Mean: {np.mean(processing_times):.2f}ms")
    print(f"  Median: {np.median(processing_times):.2f}ms")
    print(f"  Min: {np.min(processing_times):.2f}ms")
    print(f"  Max: {np.max(processing_times):.2f}ms")
    print(f"  Std: {np.std(processing_times):.2f}ms")
    
    # Check if real-time capable
    target_frame_time = 1000 / fps
    realtime_capable = np.mean(processing_times) < target_frame_time
    print(f"\nTarget frame time for {fps:.0f} FPS: {target_frame_time:.2f}ms")
    print(f"Real-time capable: {'YES ✓' if realtime_capable else 'NO ✗'}")
    
    # Show last tracking result
    if result.annotations:
        ann = result.annotations[0]
        print(f"\nLast tracked annotation:")
        print(f"  Label: {ann['label']}")
        print(f"  Position: ({ann['x']:.1f}%, {ann['y']:.1f}%)")
        print(f"  Size: {ann['width']:.1f}% x {ann['height']:.1f}%")
        print(f"  Confidence: {ann['confidence']:.2f}")
        print(f"  Deformed: {ann['deformed']}")
    
    cap.release()
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    # Default test video
    default_video = "dataset/Echo/echo1.mp4"
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    test_tracker_on_video(video_path, num_frames)
