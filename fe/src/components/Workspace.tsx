import { useState, useEffect, useRef, useCallback } from 'react';
import { OverlayCanvas } from './OverlayCanvas';
import { Settings, Brush, Play, Pause, Radio, Trash2 } from 'lucide-react';
import { useTracking } from '../hooks/useTracking';

const API_BASE = 'http://localhost:8000';

interface WorkspaceProps {
  videoPath: string | null;
}

export const Workspace = ({ videoPath }: WorkspaceProps) => {
  const [labels, setLabels] = useState<any[]>([]);
  const [saving, setSaving] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const trackingEnabled = true; // Always on
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Track if we are currently loading labels to prevent auto-save from overwriting
  const isLabelLoading = useRef(false);
  const isStartingTracking = useRef(false);
  
  // Tracking hook - updates labels with tracked positions
  const handleTrackedLabelsUpdate = useCallback((trackedLabels: any[]) => {
    // Only update if we have tracked labels
    if (trackedLabels.length > 0) {
      setLabels(trackedLabels);
    }
  }, []);
  
  const {
    isTracking,
    isConnected,
    processingTimeMs,
    startTracking,
    stopTracking,
    updateAnnotations,
    syncTime
  } = useTracking({
    videoPath,
    initialLabels: labels,
    onLabelsUpdate: handleTrackedLabelsUpdate
  });

  // Load labels
  useEffect(() => {
    if (!videoPath) {
      setLabels([]);
      return;
    }

    isLabelLoading.current = true;
    fetch(`${API_BASE}/api/labels?video_path=${encodeURIComponent(videoPath)}`)
      .then(res => res.json())
      .then(data => {
        setLabels(data);
        // Small delay to ensure the state update is processed before we allow saving again
        setTimeout(() => {
          isLabelLoading.current = false;
        }, 100);
      })
      .catch(err => {
        console.error("Failed to load labels", err);
        isLabelLoading.current = false;
      });
  }, [videoPath]);

  // Save labels function - only called on specific events, not every change
  const labelsRef = useRef(labels);
  labelsRef.current = labels;
  const videoPathRef = useRef(videoPath);
  
  const saveLabels = useCallback((labelsOverride?: any[]) => {
    const currentLabels = labelsOverride || labelsRef.current;
    const currentVideoPath = videoPathRef.current;
    if (!currentVideoPath || isLabelLoading.current) return;
    
    setSaving(true);
    fetch(`${API_BASE}/api/labels`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path: currentVideoPath, labels: currentLabels })
    })
    .then(res => res.json())
    .then(() => setSaving(false))
    .catch(err => {
      console.error("Failed to save", err);
      setSaving(false);
    });
  }, []);
  
  // Save when switching videos
  useEffect(() => {
    videoPathRef.current = videoPath;
    return () => {
      // Save previous video's labels when switching
      saveLabels();
    };
  }, [videoPath, saveLabels]);
  
  // Save on unmount
  useEffect(() => {
    return () => saveLabels();
  }, [saveLabels]);

  // Handle Play/Pause logic with tracking integration
  const togglePlay = () => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.pause();
      // Stop tracking when video pauses
      if (trackingEnabled) {
        stopTracking();
      }
    } else {
      videoRef.current.play();
      // Start tracking when video plays (if we have labels)
      if (trackingEnabled && labels.length > 0) {
        startTracking(videoRef.current.currentTime);
      }
    }
    setIsPlaying(!isPlaying);
  };
  
  const clearLabels = () => {
    if (labels.length === 0) return;
    if (window.confirm("Are you sure you want to clear all labels? This cannot be undone.")) {
        setLabels([]);
        saveLabels([]); // Save empty state immediately
    }
  };

  // Update tracking when labels change (e.g., user adds new annotation)
  useEffect(() => {
    if (isTracking && labels.length > 0) {
      const currentTime = videoRef.current?.currentTime || 0;
      updateAnnotations(labels, currentTime);
    }
  }, [labels.length]); // Only when count changes
  
  // Auto-start tracking when video is playing and labels become available
  useEffect(() => {
    if (isPlaying && labels.length > 0 && !isTracking && !isStartingTracking.current) {
      isStartingTracking.current = true;
      const currentTime = videoRef.current?.currentTime || 0;
      startTracking(currentTime).finally(() => {
        isStartingTracking.current = false;
      });
    }
  }, [isPlaying, labels.length, isTracking, startTracking]);
  
  // Note: saveLabels function removed as it is now automatic

  if (!videoPath) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center text-brand-secondary/40">
        <div className="w-24 h-24 bg-brand-primary/5 rounded-full flex items-center justify-center mb-6">
           <Brush size={40} className="text-brand-primary/40 ml-1" />
        </div>
        <p className="text-lg font-serif font-medium text-brand-secondary/80">Select a video to start labeling</p>
        <p className="text-sm font-sans text-brand-secondary/50 mt-2">Browse the dataset in the left panel</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full relative">
      
      {/* Top Bar - Floating Island Style */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 flex gap-3">
         {/* Tracking Status Pill */}
        {trackingEnabled && (
          <div className={`px-4 py-1.5 rounded-full text-xs font-serif font-semibold shadow-sm transition-all flex items-center gap-2 border
            ${isTracking 
              ? 'bg-brand-primary/10 text-brand-primary border-brand-primary/20' 
              : isConnected
                ? 'bg-white/80 text-brand-secondary/60 border-brand-primary/10'
                : 'bg-white/50 text-brand-secondary/40 border-brand-primary/5'
            }`}
          >
            <Radio size={12} className={isTracking ? 'animate-pulse' : ''} />
            <span>
              {isTracking 
                ? `Tracking ${processingTimeMs.toFixed(1)}ms` 
                : isConnected 
                  ? 'Ready' 
                  : 'Tracking Off'}
            </span>
          </div>
        )}
        
         {/* Save Status Pill */}
        <div className={`px-4 py-1.5 rounded-full text-xs font-serif font-semibold shadow-sm transition-all flex items-center gap-2 border
          ${saving 
            ? 'bg-amber-50 text-amber-700 border-amber-200' 
            : 'bg-brand-primary/10 text-brand-primary border-brand-primary/20'
          }`}
        >
          {saving ? (
             <>
               <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse"></span>
               <span>Saving...</span>
             </>
          ) : (
             <>
               <span className="w-2 h-2 rounded-full bg-brand-primary"></span>
               <span>Saved</span>
             </>
          )}
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="flex-1 flex items-center justify-center p-0 overflow-hidden relative bg-black/5 rounded-2xl">
        {/* Video Container - Responsive & Centered */}
        <div className="relative w-full h-full flex items-center justify-center group bg-black">
           <video
             ref={videoRef}
             src={`${API_BASE}/${videoPath}`}
             autoPlay
             loop
             playsInline
             className="w-full h-full object-contain"
             onPlay={(e) => {
               setIsPlaying(true);
               // Auto-start tracking when video plays
               if (trackingEnabled && labels.length > 0) {
                 startTracking((e.target as HTMLVideoElement).currentTime);
               }
             }}
             onPause={() => {
               setIsPlaying(false);
               if (trackingEnabled) {
                 stopTracking();
               }
               saveLabels(); // Save on pause
             }}
             onTimeUpdate={(e) => {
               // Sync tracker with video playback time
               if (isTracking) {
                 syncTime((e.target as HTMLVideoElement).currentTime);
               }
             }}
           />
           <OverlayCanvas 
             labels={labels} 
             onLabelsChange={setLabels} 
           />
        </div>
      </div>
      
      {/* Bottom Info Bar - Floating */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30">
        <div className="bg-white/95 backdrop-blur-md shadow-lg border border-brand-primary/10 rounded-2xl p-2 px-4 flex items-center gap-4">
             
             {/* Play/Pause Control */}
             <button 
               className="p-3 text-brand-secondary/60 hover:text-brand-primary hover:bg-brand-primary/5 rounded-xl transition-all group relative" 
               title={isPlaying ? "Pause" : "Play"}
               onClick={togglePlay}
             >
               {isPlaying ? <Pause size={20} /> : <Play size={20} />}
             </button>

             <button className="p-3 text-brand-secondary/60 hover:text-brand-primary hover:bg-brand-primary/5 rounded-xl transition-all group relative" title="Draw">
               <Brush size={20} />
               <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-brand-secondary text-white text-[10px] py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">Draw Tool</span>
             </button>
             <button className="p-3 text-brand-secondary/60 hover:text-brand-primary hover:bg-brand-primary/5 rounded-xl transition-all group relative" title="Labels">
               <div className="relative">
                 <Settings size={20} />
                 <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-primary/40 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-brand-primary text-[8px] text-white justify-center items-center">{labels.length}</span>
                  </span>
               </div>
                <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-brand-secondary text-white text-[10px] py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">Manage Labels</span>
             </button>

             <button 
                className="p-3 text-brand-secondary/60 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all group relative" 
                title="Clear Labels"
                onClick={clearLabels}
             >
               <Trash2 size={20} />
               <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-brand-secondary text-white text-[10px] py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">Clear All</span>
             </button>
             
             <div className="flex flex-col px-3 min-w-[120px] border-l border-brand-primary/10 pl-6 ml-2">
               <span className="text-[10px] font-bold text-brand-secondary/40 uppercase tracking-widest font-serif">Active Video</span>
               <span className="text-xs font-semibold text-brand-secondary truncate max-w-[160px] font-sans">{videoPath.split('/').pop()}</span>
             </div>
        </div>
      </div>
      
    </div>
  );
};
