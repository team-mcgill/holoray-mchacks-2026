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
  // Freeform drawn points (percentage coordinates)
  points?: [number, number][];
  // Tracking fields
  confidence?: number;
  deformed?: boolean;
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
    console.log('[Tracking] startSession called with', currentLabels.length, 'labels, video:', videoPath);
    
    if (!videoPath || currentLabels.length === 0) {
      console.log('[Tracking] Cannot start session: no video or no labels');
      return null;
    }

    try {
      console.log('[Tracking] Creating session with annotations:', currentLabels);
      const response = await fetch(`${API_BASE}/api/tracking/start?video_path=${encodeURIComponent(videoPath)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(currentLabels)
      });

      if (!response.ok) {
        throw new Error('Failed to start tracking session');
      }

      const data = await response.json();
      console.log('[Tracking] Session created:', data.session_id);
      setSessionId(data.session_id);
      return data.session_id;
    } catch (error) {
      console.error('Failed to start tracking session:', error);
      return null;
    }
  }, [videoPath]);

  // Connect to WebSocket for real-time updates
  const connect = useCallback(async (labels?: TrackedLabel[]): Promise<boolean> => {
    let sid = sessionId;
    if (!sid) {
      sid = await startSession(labels);
      if (!sid) {
        console.log('[Tracking] No session ID, cannot connect');
        return false;
      }
    }

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    console.log('[Tracking] Connecting WebSocket to', `${WS_BASE}/ws/tracking/${sid}`);
    const ws = new WebSocket(`${WS_BASE}/ws/tracking/${sid}`);
    wsRef.current = ws;
    
    return new Promise((resolve) => {
      ws.onopen = () => {
        setState(prev => ({ ...prev, isConnected: true }));
        console.log('[Tracking] WebSocket connected!');
        resolve(true);
      };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('[Tracking] Received:', data.frame_idx !== undefined ? `frame ${data.frame_idx}` : data);
        
        if (data.error) {
          console.error('[Tracking] Error:', data.error);
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
      console.error('[Tracking] WebSocket error:', error);
      resolve(false);
    };
    });
  }, [sessionId, startSession, onLabelsUpdate]);

  // Start tracking
  const startTracking = useCallback(async () => {
    const currentLabels = labelsRef.current;
    console.log('[Tracking] startTracking called with', currentLabels.length, 'labels');
    
    if (currentLabels.length === 0) {
      console.log('[Tracking] No labels to track');
      return;
    }
    
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      const connected = await connect(currentLabels);
      if (!connected) {
        console.log('[Tracking] Failed to connect');
        return;
      }
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[Tracking] Sending start action');
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
  
  // Sync tracking with video time (call this on video timeupdate)
  const syncTime = useCallback((currentTime: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        action: 'sync',
        time: currentTime
      }));
    }
  }, []);

  // Reset session when video changes
  useEffect(() => {
    setSessionId(null);
    setState(prev => ({ ...prev, isTracking: false, isConnected: false }));
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, [videoPath]);

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
    syncTime,
    connect
  };
}
