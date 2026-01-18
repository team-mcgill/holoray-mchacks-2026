"""
Utility functions for deformable tracking.
"""
import numpy as np
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional

def sample_boundary_points(
    bbox: dict, 
    num_points: int = 32,
    shape_type: str = "rectangle"
) -> np.ndarray:
    """
    Sample points along the boundary of an annotation.
    
    Args:
        bbox: Dictionary with x, y, width, height (in percentages 0-100)
        num_points: Number of points to sample along boundary
        shape_type: "rectangle" or "ellipse"
    
    Returns:
        np.ndarray of shape (num_points, 2) with (x, y) coordinates
    """
    x, y = bbox['x'], bbox['y']
    w, h = bbox['width'], bbox['height']
    
    if shape_type == "ellipse":
        # Sample points along ellipse
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        cx, cy = x + w/2, y + h/2
        rx, ry = w/2, h/2
        points = np.column_stack([
            cx + rx * np.cos(angles),
            cy + ry * np.sin(angles)
        ])
    else:
        # Sample points along rectangle perimeter
        perimeter = 2 * (w + h)
        points_per_side = {
            'top': max(1, int(num_points * w / perimeter)),
            'right': max(1, int(num_points * h / perimeter)),
            'bottom': max(1, int(num_points * w / perimeter)),
            'left': max(1, int(num_points * h / perimeter))
        }
        
        # Adjust to match num_points exactly
        total = sum(points_per_side.values())
        if total < num_points:
            points_per_side['top'] += num_points - total
        
        points = []
        # Top edge
        for i in range(points_per_side['top']):
            t = i / max(1, points_per_side['top'] - 1) if points_per_side['top'] > 1 else 0
            points.append([x + t * w, y])
        # Right edge
        for i in range(points_per_side['right']):
            t = i / max(1, points_per_side['right'] - 1) if points_per_side['right'] > 1 else 0
            points.append([x + w, y + t * h])
        # Bottom edge
        for i in range(points_per_side['bottom']):
            t = 1 - i / max(1, points_per_side['bottom'] - 1) if points_per_side['bottom'] > 1 else 1
            points.append([x + t * w, y + h])
        # Left edge
        for i in range(points_per_side['left']):
            t = 1 - i / max(1, points_per_side['left'] - 1) if points_per_side['left'] > 1 else 1
            points.append([x, y + t * h])
        
        points = np.array(points[:num_points])
    
    return points


def reconstruct_contour(
    points: np.ndarray,
    smoothing: float = 0.0,
    num_output_points: int = 100
) -> np.ndarray:
    """
    Reconstruct a smooth contour from tracked boundary points using spline interpolation.
    
    Args:
        points: np.ndarray of shape (N, 2) with tracked point coordinates
        smoothing: Smoothing factor for spline (0 = interpolating spline)
        num_output_points: Number of points in output contour
    
    Returns:
        np.ndarray of shape (num_output_points, 2) with smooth contour
    """
    if len(points) < 4:
        return points
    
    # Close the contour by appending first point
    closed_points = np.vstack([points, points[0:1]])
    
    try:
        # Fit periodic spline
        tck, u = splprep([closed_points[:, 0], closed_points[:, 1]], 
                         s=smoothing, per=True)
        
        # Evaluate spline at uniform intervals
        u_new = np.linspace(0, 1, num_output_points)
        x_new, y_new = splev(u_new, tck)
        
        return np.column_stack([x_new, y_new])
    except Exception:
        # Fallback: return original points if spline fails
        return points


def compute_bbox_from_points(points: np.ndarray) -> dict:
    """
    Compute bounding box from a set of points.
    
    Args:
        points: np.ndarray of shape (N, 2)
    
    Returns:
        dict with x, y, width, height
    """
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    return {
        'x': float(x_min),
        'y': float(y_min),
        'width': float(x_max - x_min),
        'height': float(y_max - y_min)
    }


def points_to_svg_path(points: np.ndarray, closed: bool = True) -> str:
    """
    Convert points to SVG path string for frontend rendering.
    
    Args:
        points: np.ndarray of shape (N, 2)
        closed: Whether to close the path
    
    Returns:
        SVG path string (e.g., "M 10 20 L 30 40 L 50 60 Z")
    """
    if len(points) == 0:
        return ""
    
    path_parts = [f"M {points[0, 0]:.2f} {points[0, 1]:.2f}"]
    
    for i in range(1, len(points)):
        path_parts.append(f"L {points[i, 0]:.2f} {points[i, 1]:.2f}")
    
    if closed:
        path_parts.append("Z")
    
    return " ".join(path_parts)
