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
          className="flex items-center gap-2 py-1.5 hover:bg-brand-primary/5 cursor-pointer text-brand-secondary/80 transition-colors rounded-sm mx-2"
          style={{ paddingLeft }}
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <Folder size={14} className="text-brand-primary/40" />
          <span className="text-sm font-sans font-medium truncate">{node.name}</span>
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
      className={`flex items-center gap-2 py-1.5 cursor-pointer transition-colors rounded-sm mx-2
        ${isSelected 
          ? 'bg-brand-primary/10 text-brand-primary font-medium' 
          : 'text-brand-secondary/70 hover:bg-brand-primary/5'
        }`}
      style={{ paddingLeft }}
      onClick={() => onSelect(node.path)}
    >
      <FileVideo size={14} className={isSelected ? 'text-brand-primary' : 'opacity-70'} />
      <span className="text-sm font-sans truncate">{node.name}</span>
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
    <div className="h-full flex flex-col bg-transparent overflow-hidden">
      {/* Header */}
      <div className="pb-4">
        <h2 className="font-serif font-semibold text-brand-secondary text-lg mb-4 pl-1">Assets</h2>
        
        {/* Search / Filter (Visual only for now) */}
        <div className="relative group">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-brand-secondary/40 group-focus-within:text-brand-primary transition-colors" />
          <input 
            type="text" 
            placeholder="Search..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white/40 border border-brand-primary/10 focus:border-brand-primary/30 focus:bg-white/60 text-brand-secondary text-sm rounded-sm pl-9 pr-3 py-2 outline-none transition-all placeholder:text-brand-secondary/30"
          />
        </div>
      </div>

      {/* Tree */}
      <div className="flex-1 overflow-y-auto py-1 custom-scrollbar -ml-2">
        {loading ? (
          <div className="p-4 text-sm text-brand-secondary/40 text-center animate-pulse font-serif italic">Loading dataset...</div>
        ) : filteredTree.length === 0 ? (
          <div className="p-4 text-sm text-brand-secondary/40 text-center font-serif italic">No assets found</div>
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
      <div className="pt-4 border-t border-brand-primary/10 text-[10px] text-brand-secondary/40 flex justify-between font-serif uppercase tracking-widest">
        <span>Dataset v1.0</span>
        <span>{tree.length} Root Items</span>
      </div>
    </div>
  );
};
