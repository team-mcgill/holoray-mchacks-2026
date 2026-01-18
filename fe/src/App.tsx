import { useState } from 'react';
import { VideoList } from './components/VideoList';
import { Workspace } from './components/Workspace';

function App() {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);

  return (
    <div className="flex h-screen w-screen overflow-hidden text-slate-900 font-sans bg-[#f3f4f6]">
      {/* Sidebar - Floating Card */}
      <div className="w-[280px] p-4 pr-0 flex flex-col h-full">
         <div className="mb-4 pl-2 pt-2">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center text-white font-bold">H</div>
              <h1 className="font-bold text-xl text-slate-800 tracking-tight">Holoray</h1>
            </div>
         </div>
         <VideoList 
           onSelectVideo={setSelectedVideo} 
           selectedVideo={selectedVideo} 
         />
      </div>
      
      {/* Main Workspace */}
      <div className="flex-1 h-full p-4">
        <div className="h-full w-full bg-white/50 rounded-3xl shadow-sm border border-white/40 backdrop-blur-sm overflow-hidden flex flex-col workspace-grid relative">
           <Workspace videoPath={selectedVideo} />
        </div>
      </div>
    </div>
  );
}

export default App;
