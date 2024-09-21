
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict
import cv2


class PredictionsSink:
    def __init__(
        self,
        target_filepath: Path
    ):        
        self.target_filepath = target_filepath        
        def new_item():
            return []
        self.data = defaultdict(new_item) 
        self.count = 0      


    def write(self, item: Dict):
        self.data["item"].append(self.count)        
        for k,v in item.items():            
            if ("image" in k):
                self.write_image(self.count, k, v)
            else:
                self.data[k].append(v)
                
        self.count += 1
            
            
    def flush(self):
        df = pd.DataFrame(self.data)
        self.target_filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.target_filepath)        
        
    def write_image(self, idx, name, image):
        folder = self.target_filepath.parent / "images" / self.target_filepath.stem / name
        filepath = folder / f"{idx:06d}.png"        
        filepath.parent.mkdir(parents=True, exist_ok=True)        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath.as_posix(), image)