import { useState, useEffect, useRef } from 'react';
import { OverlayCanvas } from './OverlayCanvas';
import { Settings, MousePointer2, Play, Pause } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

interface WorkspaceProps {
  videoPath: string | null;
}

export const Workspace = ({ videoPath }: WorkspaceProps) => {
  const [labels, setLabels] = useState<any[]>([]);
  const [saving, setSaving] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  
  // Track if we are currently loading labels to prevent auto-save from overwriting
  const isLabelLoading = useRef(false);

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

  // Auto-save labels whenever they change
  useEffect(() => {
    // Prevent saving if no video, or if we are currently loading initial labels for a new video
    if (!videoPath || isLabelLoading.current) return;
    
    // Debounce slightly to avoid spamming while drawing? 
    // For now, let's just save. Optimistic UI updates happen in OverlayCanvas.
    
    setSaving(true);
    fetch(`${API_BASE}/api/labels`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path: videoPath, labels })
    })
    .then(res => res.json())
    .then(() => {
      setSaving(false);
    })
    .catch(err => {
      console.error("Failed to auto-save", err);
      setSaving(false);
    });

  }, [labels, videoPath]); // Trigger on labels change

  // Handle Play/Pause logic
  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };
  
  // Note: saveLabels function removed as it is now automatic

  if (!videoPath) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-300">
        <div className="w-24 h-24 bg-slate-100 rounded-full flex items-center justify-center mb-6">
           <MousePointer2 size={40} className="text-slate-300 ml-1" />
        </div>
        <p className="text-lg font-medium text-slate-500">Select a video to start labeling</p>
        <p className="text-sm text-slate-400 mt-2">Browse the dataset in the left panel</p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full relative">
      
      {/* Top Bar - Floating Island Style */}
      <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 flex gap-4">
         {/* Status Pill */}
        <div className={`px-4 py-1.5 rounded-full text-xs font-semibold shadow-sm transition-all flex items-center gap-2 border
          ${saving 
            ? 'bg-yellow-50 text-yellow-600 border-yellow-200' 
            : 'bg-green-50 text-green-600 border-green-200'
          }`}
        >
          {saving ? (
             <>
               <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></span>
               <span>Saving...</span>
             </>
          ) : (
             <>
               <span className="w-2 h-2 rounded-full bg-green-500"></span>
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
             onPlay={() => setIsPlaying(true)}
             onPause={() => setIsPlaying(false)}
             // Hide default controls to use ours, but allow right-click controls if needed
           />
           <OverlayCanvas 
             labels={labels} 
             onLabelsChange={setLabels} 
           />
        </div>
      </div>
      
      {/* Bottom Info Bar - Floating */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30">
        <div className="bg-white/90 backdrop-blur-md shadow-lg border border-white/20 rounded-2xl p-2 flex items-center gap-1">
             
             {/* Play/Pause Control */}
             <button 
               className="p-3 text-slate-500 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-all group relative" 
               title={isPlaying ? "Pause" : "Play"}
               onClick={togglePlay}
             >
               {isPlaying ? <Pause size={20} /> : <Play size={20} />}
             </button>

             <div className="w-px h-8 bg-slate-200 mx-2"></div>

             <button className="p-3 text-slate-500 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-all group relative" title="Select">
               <MousePointer2 size={20} />
               <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-[10px] py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">Select Tool</span>
             </button>
             <button className="p-3 text-slate-500 hover:text-blue-600 hover:bg-blue-50 rounded-xl transition-all group relative" title="Labels">
               <div className="relative">
                 <Settings size={20} />
                 <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500 text-[8px] text-white justify-center items-center">{labels.length}</span>
                  </span>
               </div>
                <span className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-[10px] py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap">Manage Labels</span>
             </button>
             
             <div className="w-px h-8 bg-slate-200 mx-2"></div>
             
             <div className="flex flex-col px-3 min-w-[120px]">
               <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Active Video</span>
               <span className="text-xs font-semibold text-slate-700 truncate max-w-[160px]">{videoPath.split('/').pop()}</span>
             </div>
        </div>
      </div>
      
    </div>
  );
};
