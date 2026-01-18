from pydantic import BaseModel
from typing import List, Optional

class Label(BaseModel):
    id: str
    label: str
    x: float
    y: float
    width: float
    height: float
    color: str = "#FF0000"

class VideoLabels(BaseModel):
    video_path: str
    labels: List[Label]
