import { useRef, useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { X } from 'lucide-react';

interface Label {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  // Polygon/freeform points (percentage coordinates)
  points?: [number, number][];
  // Deformable tracking fields (optional - present when tracked)
  deformed?: boolean;
  svg_path?: string;
  contour_points?: [number, number][];
  confidence?: number;
}

type DrawMode = 'box' | 'draw';

interface OverlayCanvasProps {
  labels: Label[];
  onLabelsChange: (labels: Label[]) => void;
  isTracking?: boolean;
  drawMode?: DrawMode;
}

export const OverlayCanvas = ({ labels, onLabelsChange, drawMode = 'draw' }: OverlayCanvasProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentStart, setCurrentStart] = useState<{ x: number, y: number } | null>(null);
  const [currentBox, setCurrentBox] = useState<Partial<Label> | null>(null);
  const [currentPoints, setCurrentPoints] = useState<[number, number][]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  
  // Helper to convert points to SVG path
  const pointsToSvgPath = (pts: [number, number][]) => {
    if (pts.length < 2) return '';
    return `M ${pts.map(p => `${p[0]} ${p[1]}`).join(' L ')} Z`;
  };
  
  // Helper to get bounding box from points
  const getBoundingBox = (pts: [number, number][]) => {
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    // If we are clicking on an input/button, don't start drawing
    if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'BUTTON') return;

    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;

    setIsDrawing(true);
    setCurrentStart({ x, y });
    setEditingId(null);
    
    if (drawMode === 'draw') {
      setCurrentPoints([[x, y]]);
    } else {
      setCurrentBox({ x, y, width: 0, height: 0 });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const currentX = ((e.clientX - rect.left) / rect.width) * 100;
    const currentY = ((e.clientY - rect.top) / rect.height) * 100;

    if (drawMode === 'draw') {
      // Add point every few pixels for smoother drawing
      const lastPoint = currentPoints[currentPoints.length - 1];
      const dist = Math.sqrt((currentX - lastPoint[0]) ** 2 + (currentY - lastPoint[1]) ** 2);
      if (dist > 0.5) { // Add point every 0.5% distance
        setCurrentPoints([...currentPoints, [currentX, currentY]]);
      }
    } else {
      if (!currentStart) return;
      const width = Math.abs(currentX - currentStart.x);
      const height = Math.abs(currentY - currentStart.y);
      const x = Math.min(currentX, currentStart.x);
      const y = Math.min(currentY, currentStart.y);
      setCurrentBox({ x, y, width, height });
    }
  };

  const handleMouseUp = () => {
    if (drawMode === 'draw' && isDrawing && currentPoints.length > 5) {
      // Simplify points - keep every Nth point
      const simplified = currentPoints.filter((_, i) => i % 3 === 0 || i === currentPoints.length - 1);
      const bbox = getBoundingBox(simplified);
      
      if (bbox.width > 1 && bbox.height > 1) {
        const newLabel: Label = {
          id: uuidv4(),
          label: "New Label",
          ...bbox,
          points: simplified,
          color: "#0ea5e9"
        };
        onLabelsChange([...labels, newLabel]);
        setEditingId(newLabel.id);
      }
    } else if (drawMode === 'box' && isDrawing && currentBox && currentBox.width && currentBox.width > 1 && currentBox.height && currentBox.height > 1) {
      const newLabel: Label = {
        id: uuidv4(),
        label: "New Label",
        x: currentBox.x!,
        y: currentBox.y!,
        width: currentBox.width!,
        height: currentBox.height!,
        color: "#0ea5e9"
      };
      onLabelsChange([...labels, newLabel]);
      setEditingId(newLabel.id);
    }
    
    setIsDrawing(false);
    setCurrentBox(null);
    setCurrentStart(null);
    setCurrentPoints([]);
  };

  const updateLabelName = (id: string, name: string) => {
    const updated = labels.map(l => l.id === id ? { ...l, label: name } : l);
    onLabelsChange(updated);
  };
  
  const deleteLabel = (id: string) => {
    onLabelsChange(labels.filter(l => l.id !== id));
  };

  return (
    <div
      ref={containerRef}
      className="absolute inset-0 z-10 cursor-crosshair"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      // When clicking background, deselect current label
      onClick={() => setEditingId(null)} 
    >
      {/* Existing Labels */}
      {labels.map(label => {
        // Check if this is a deformed (tracked) annotation with SVG path
        const isDeformed = label.deformed && label.svg_path;
        
        if (isDeformed) {
          // Render deformed annotation using SVG path
          return (
            <svg
              key={label.id}
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <path
                d={label.svg_path}
                fill={`${label.color}15`}
                stroke={label.color}
                strokeWidth="0.3"
                className="pointer-events-auto cursor-pointer hover:stroke-blue-400 transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingId(label.id);
                }}
              />
              {/* Label text positioned at centroid */}
              <text
                x={label.x + label.width / 2}
                y={label.y - 1}
                fill="white"
                fontSize="2"
                textAnchor="middle"
                className="pointer-events-none select-none"
                style={{ textShadow: '0 0 2px black' }}
              >
                {label.label}
              </text>
              {/* Confidence indicator */}
              {label.confidence !== undefined && (
                <text
                  x={label.x + label.width / 2}
                  y={label.y + label.height + 2}
                  fill={label.confidence > 0.7 ? '#22c55e' : label.confidence > 0.4 ? '#eab308' : '#ef4444'}
                  fontSize="1.5"
                  textAnchor="middle"
                  className="pointer-events-none select-none"
                >
                  {Math.round(label.confidence * 100)}%
                </text>
              )}
            </svg>
          );
        }
        
        // Check if this has polygon points
        if (label.points && label.points.length > 2) {
          const path = pointsToSvgPath(label.points);
          return (
            <svg
              key={label.id}
              className="absolute inset-0 w-full h-full pointer-events-none"
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <path
                d={path}
                fill={`${label.color}20`}
                stroke={editingId === label.id ? '#0ea5e9' : label.color}
                strokeWidth="0.3"
                className="pointer-events-auto cursor-pointer hover:stroke-blue-400 transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  setEditingId(label.id);
                }}
              />
              {/* Label text */}
              <text
                x={label.x + label.width / 2}
                y={label.y - 1}
                fill="white"
                fontSize="2"
                textAnchor="middle"
                className="pointer-events-none select-none"
                style={{ textShadow: '0 0 2px black' }}
              >
                {label.label}
              </text>
              {/* Editor popup for polygon */}
              {editingId === label.id && (
                <foreignObject x={label.x} y={label.y - 8} width="40" height="6">
                  <div className="bg-white shadow-lg text-xs px-2 py-1 rounded-lg border border-slate-200 flex items-center gap-2 whitespace-nowrap">
                    <input 
                      className="outline-none bg-transparent w-16 font-medium text-slate-700 text-[10px]"
                      value={label.label}
                      onChange={(e) => updateLabelName(label.id, e.target.value)}
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => { if (e.key === 'Enter') setEditingId(null); }}
                    />
                    <button 
                      className="text-slate-400 hover:text-red-500 transition-colors"
                      onClick={(e) => { e.stopPropagation(); deleteLabel(label.id); }}
                    >
                      <X size={10} />
                    </button>
                  </div>
                </foreignObject>
              )}
            </svg>
          );
        }
        
        // Regular rectangle annotation (fallback)
        return (
          <div
            key={label.id}
            className="absolute border-2 pointer-events-auto group hover:border-blue-400 transition-colors"
            style={{
              left: `${label.x}%`,
              top: `${label.y}%`,
              width: `${label.width}%`,
              height: `${label.height}%`,
              borderColor: editingId === label.id ? '#0ea5e9' : label.color,
              backgroundColor: `${label.color}10`
            }}
            onClick={(e) => {
              e.stopPropagation();
              setEditingId(label.id);
            }}
          >
            {/* Editor (Input) - Visible when selected */}
            {editingId === label.id && (
              <div 
                 className="absolute -top-9 left-0 bg-white shadow-lg text-xs px-2 py-1.5 rounded-lg border border-slate-200 flex items-center gap-2 whitespace-nowrap z-20 animate-in fade-in zoom-in-95 duration-100"
                 onClick={(e) => e.stopPropagation()}
              >
                <input 
                  className="outline-none bg-transparent w-24 font-medium text-slate-700 placeholder:text-slate-400"
                  value={label.label}
                  onChange={(e) => updateLabelName(label.id, e.target.value)}
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') setEditingId(null);
                  }}
                />
                 <button 
                  className="text-slate-400 hover:text-red-500 transition-colors p-0.5 rounded hover:bg-slate-100"
                  onClick={() => deleteLabel(label.id)}
                >
                  <X size={14} />
                </button>
              </div>
            )}
            
            {/* Hover Tag (when not editing) */}
            {editingId !== label.id && (
               <div 
                 className="absolute -top-7 left-0 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity z-20"
              >
                <span className="bg-black/75 backdrop-blur-sm text-white text-[10px] px-2 py-1 rounded-md whitespace-nowrap font-medium shadow-sm">
                  {label.label}
                </span>
                <button 
                  className="bg-red-500 text-white p-1 rounded-md shadow-sm hover:bg-red-600 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteLabel(label.id);
                  }}
                >
                  <X size={10} />
                </button>
              </div>
            )}
          </div>
        );
      })}

      {/* Drawing Preview - Freeform */}
      {isDrawing && drawMode === 'draw' && currentPoints.length > 1 && (
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <path
            d={`M ${currentPoints.map(p => `${p[0]} ${p[1]}`).join(' L ')}`}
            fill="none"
            stroke="#0ea5e9"
            strokeWidth="0.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )}
      
      {/* Drawing Preview - Box */}
      {isDrawing && drawMode === 'box' && currentBox && (
        <div
          className="absolute border-2 border-blue-400 bg-blue-400/20"
          style={{
            left: `${currentBox.x}%`,
            top: `${currentBox.y}%`,
            width: `${currentBox.width}%`,
            height: `${currentBox.height}%`,
          }}
        />
      )}
    </div>
  );
};
