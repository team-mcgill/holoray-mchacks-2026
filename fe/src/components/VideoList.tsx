import { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, Folder, FileVideo, Search } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

interface FileNode {
  name: string;
  type: 'folder' | 'file';
  path: string;
  children?: FileNode[];
}

interface VideoListProps {
  onSelectVideo: (path: string) => void;
  selectedVideo: string | null;
}

const FileTreeItem = ({ node, level, onSelect, selectedPath, defaultOpen }: { node: FileNode, level: number, onSelect: (path: string) => void, selectedPath: string | null, defaultOpen?: boolean }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen || false);
  
  // Update open state if defaultOpen changes (e.g. when searching)
  useEffect(() => {
    if (defaultOpen !== undefined) {
      setIsOpen(defaultOpen);
    }
  }, [defaultOpen]);
  
  const isSelected = node.path === selectedPath;
  const paddingLeft = `${level * 16 + 12}px`;

  if (node.type === 'folder') {
    return (
      <div className="select-none">
        <div 
          className="flex items-center gap-2 py-1.5 hover:bg-slate-100 cursor-pointer text-slate-600 transition-colors rounded-md mx-2"
          style={{ paddingLeft }}
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Folder size={14} className="text-slate-400" />
          <span className="text-sm font-medium truncate">{node.name}</span>
        </div>
        {isOpen && node.children && (
          <div>
            {node.children.map((child) => (
              <FileTreeItem 
                key={child.path} 
                node={child} 
                level={level + 1} 
                onSelect={onSelect}
                selectedPath={selectedPath}
                defaultOpen={defaultOpen}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div 
      className={`flex items-center gap-2 py-1.5 cursor-pointer transition-colors rounded-md mx-2
        ${isSelected 
          ? 'bg-blue-50 text-blue-600' 
          : 'text-slate-500 hover:bg-slate-100'
        }`}
      style={{ paddingLeft }}
      onClick={() => onSelect(node.path)}
    >
      <FileVideo size={14} />
      <span className="text-sm truncate">{node.name}</span>
    </div>
  );
};

export const VideoList = ({ onSelectVideo, selectedVideo }: VideoListProps) => {
  const [tree, setTree] = useState<FileNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/api/tree`)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
            setTree(data);
        } else {
            console.error("API returned non-array for tree:", data);
            setTree([]);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to fetch video tree", err);
        setTree([]); // Ensure tree is an array on error
        setLoading(false);
      });
  }, []);

  // Filter tree based on search query
  const filterTree = (nodes: FileNode[], query: string): FileNode[] => {
    if (!query) return nodes;

    return nodes.reduce((acc: FileNode[], node) => {
      // If folder, check children recursively
      if (node.type === 'folder' && node.children) {
        const filteredChildren = filterTree(node.children, query);
        // Include folder if it has matching children OR if the folder name matches
        if (filteredChildren.length > 0 || node.name.toLowerCase().includes(query.toLowerCase())) {
          acc.push({ ...node, children: filteredChildren });
        }
      } else if (node.type === 'file') {
        // If file, check name
        if (node.name.toLowerCase().includes(query.toLowerCase())) {
          acc.push(node);
        }
      }
      return acc;
    }, []);
  };

  const filteredTree = filterTree(tree, searchQuery);

  return (
    <div className="h-full flex flex-col bg-white/50 backdrop-blur-xl rounded-2xl shadow-lg border border-white/20 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-100">
        <h2 className="font-semibold text-slate-800 text-lg mb-4">Assets</h2>
        
        {/* Search / Filter (Visual only for now) */}
        <div className="relative group">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-blue-500 transition-colors" />
          <input 
            type="text" 
            placeholder="Search..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-50 border border-transparent focus:border-blue-200 focus:bg-white text-sm rounded-xl pl-9 pr-3 py-2 outline-none transition-all"
          />
        </div>
      </div>

      {/* Tree */}
      <div className="flex-1 overflow-y-auto py-3 custom-scrollbar">
        {loading ? (
          <div className="p-4 text-sm text-slate-400 text-center animate-pulse">Loading dataset...</div>
        ) : filteredTree.length === 0 ? (
          <div className="p-4 text-sm text-slate-400 text-center">No assets found</div>
        ) : (
          filteredTree.map((node) => (
            <FileTreeItem 
              key={node.path} 
              node={node} 
              level={0} 
              onSelect={onSelectVideo}
              selectedPath={selectedVideo}
              defaultOpen={!!searchQuery} // Auto-expand if searching
            />
          ))
        )}
      </div>
      
      {/* Footer / Stats */}
      <div className="p-3 bg-slate-50/50 border-t border-slate-100 text-[10px] text-slate-400 flex justify-between">
        <span>Dataset v1.0</span>
        <span>{tree.length} Root Items</span>
      </div>
    </div>
  );
};
