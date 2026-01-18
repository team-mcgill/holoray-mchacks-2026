/**
 * React hook for real-time deformable annotation tracking.
 * Connects to backend WebSocket for live tracking updates.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

interface TrackingState {
  isTracking: boolean;
  isConnected: boolean;
  currentFrame: number;
  totalFrames: number;
  fps: number;
  processingTimeMs: number;
  methodUsed: string;
}

interface TrackedLabel {
  id: string;
  label: string;
  color: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  deformed: boolean;
  svg_path?: string;
  contour_points?: [number, number][];
}

interface UseTrackingOptions {
  videoPath: string | null;
  initialLabels: TrackedLabel[];
  onLabelsUpdate?: (labels: TrackedLabel[]) => void;
}

export function useTracking({ videoPath, initialLabels, onLabelsUpdate }: UseTrackingOptions) {
  const [state, setState] = useState<TrackingState>({
    isTracking: false,
    isConnected: false,
    currentFrame: 0,
    totalFrames: 0,
    fps: 30,
    processingTimeMs: 0,
    methodUsed: 'none'
  });
  
  const [trackedLabels, setTrackedLabels] = useState<TrackedLabel[]>(initialLabels);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  // Keep a ref to current labels so we always have the latest
  const labelsRef = useRef<TrackedLabel[]>(initialLabels);
  labelsRef.current = initialLabels;

  // Start tracking session with current labels
  const startSession = useCallback(async (labels?: TrackedLabel[]) => {
    const currentLabels = labels || labelsRef.current;
    if (!videoPath || currentLabels.length === 0) {
      console.log('Cannot start session: no video or no labels');
      return null;
    }

    try {
      const response = await fetch(`${API_BASE}/api/tracking/start?video_path=${encodeURIComponent(videoPath)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentLabels)
      });

      if (!response.ok) {
        throw new Error('Failed to start tracking session');
      }

      const data = await response.json();
      setSessionId(data.session_id);
      return data.session_id;
    } catch (error) {
      console.error('Failed to start tracking session:', error);
      return null;
    }
  }, [videoPath]);

  // Connect to WebSocket for real-time updates
  const connect = useCallback(async (labels?: TrackedLabel[]) => {
    let sid = sessionId;
    if (!sid) {
      sid = await startSession(labels);
      if (!sid) return;
    }

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(`${WS_BASE}/ws/tracking/${sid}`);
    
    ws.onopen = () => {
      setState(prev => ({ ...prev, isConnected: true }));
      console.log('Tracking WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.error) {
          console.error('Tracking error:', data.error);
          return;
        }

        if (data.event === 'video_ended') {
          setState(prev => ({ ...prev, isTracking: false }));
          return;
        }

        // Update state with tracking results
        setState(prev => ({
          ...prev,
          currentFrame: data.frame_idx || prev.currentFrame,
          totalFrames: data.total_frames || prev.totalFrames,
          fps: data.fps || prev.fps,
          processingTimeMs: data.processing_time_ms || prev.processingTimeMs,
          methodUsed: data.method_used || prev.methodUsed
        }));

        // Update tracked labels
        if (data.annotations) {
          const newLabels = data.annotations.map((ann: any) => ({
            ...ann,
            deformed: ann.deformed || false
          }));
          setTrackedLabels(newLabels);
          onLabelsUpdate?.(newLabels);
        }
      } catch (e) {
        console.error('Failed to parse tracking message:', e);
      }
    };

    ws.onclose = () => {
      setState(prev => ({ ...prev, isConnected: false, isTracking: false }));
      console.log('Tracking WebSocket closed');
      
      // Attempt reconnect after delay
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };

    ws.onerror = (error) => {
      console.error('Tracking WebSocket error:', error);
    };

    wsRef.current = ws;
  }, [sessionId, startSession, onLabelsUpdate]);

  // Start tracking
  const startTracking = useCallback(() => {
    const currentLabels = labelsRef.current;
    console.log('Starting tracking with', currentLabels.length, 'labels');
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect(currentLabels).then(() => {
        // Wait for connection then start
        setTimeout(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: 'start' }));
            setState(prev => ({ ...prev, isTracking: true }));
          }
        }, 100);
      });
    } else {
      wsRef.current.send(JSON.stringify({ action: 'start' }));
      setState(prev => ({ ...prev, isTracking: true }));
    }
  }, [connect]);

  // Stop tracking
  const stopTracking = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'stop' }));
    }
    setState(prev => ({ ...prev, isTracking: false }));
  }, []);

  // Update annotations being tracked
  const updateAnnotations = useCallback((labels: TrackedLabel[]) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'update_annotations',
        annotations: labels
      }));
    }
  }, []);

  // Seek to frame
  const seekToFrame = useCallback((frameIdx: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'seek',
        frame: frameIdx
      }));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      // Stop session
      if (sessionId) {
        fetch(`${API_BASE}/api/tracking/stop/${sessionId}`, { method: 'POST' })
          .catch(() => {});
      }
    };
  }, [sessionId]);

  return {
    ...state,
    trackedLabels,
    startTracking,
    stopTracking,
    updateAnnotations,
    seekToFrame,
    connect
  };
}
