import { useState } from 'react';
import { VideoList } from './components/VideoList';
import { Workspace } from './components/Workspace';

function App() {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);

  return (
    <div className="flex h-screen w-screen overflow-hidden text-brand-secondary font-sans bg-bg-paper">
      {/* Sidebar */}
      <div className="w-[320px] p-6 border-r border-brand-primary/10 flex flex-col h-full bg-bg-paper">
         <div className="mb-8">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-brand-primary rounded-sm flex items-center justify-center text-white font-serif font-bold text-xl shadow-sm">L</div>
              <h1 className="font-serif font-bold text-3xl text-brand-primary tracking-tight">Liquid</h1>
            </div>
         </div>
         <VideoList 
           onSelectVideo={setSelectedVideo} 
           selectedVideo={selectedVideo} 
         />
      </div>
      
      {/* Main Workspace */}
      <div className="flex-1 h-full relative flex flex-col workspace-grid bg-bg-paper">
         <div className="h-full w-full flex flex-col p-6">
            <div className="h-full w-full rounded-sm border border-brand-primary/5 bg-white/40 shadow-sm backdrop-blur-[2px] overflow-hidden relative">
               <Workspace videoPath={selectedVideo} />
            </div>
         </div>
      </div>
    </div>
  );
}

export default App;
