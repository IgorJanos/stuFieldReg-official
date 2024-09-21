
from pathlib import Path
from collections import defaultdict

import pandas as pd


class BaseEvaluationSink:
    def __init__(
        self,
        target_filepath: Path
    ):
        self.target_filepath = target_filepath
        def new_item():
            return []
        self.data = defaultdict(new_item)       


    def flush(self):
        df = self.build_dataframe()
        self.target_filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.target_filepath)  
        return df  
        
    def build_dataframe(self):
        return pd.DataFrame({            
        })
        
        
class EvaluationSink_IOU(BaseEvaluationSink):
    def __init__(self, target_filepath: Path):
        super().__init__(target_filepath)
        
    def write(self, item: dict):       
        # Write to CSV
        for k,v in item.items():
            self.data[k].append(v)

    def build_dataframe(self):
        return pd.DataFrame({
            "name": self.data["name"],
            "success": self.data["success"],
            "iou_part": self.data["iou_part"],            
        })
        
        
class EvaluationSink_ACt(BaseEvaluationSink):
    def __init__(self, target_filepath: Path):
        super().__init__(target_filepath)
        
    def write(self, item: dict):       
        # Write to CSV
        for k,v in item.items():
            self.data[k].append(v)

    def build_dataframe(self):
        return pd.DataFrame({
            "name": self.data["name"],
            "success": self.data["success"],
            "ac@5": self.data["ac@5"],
            "ac@10": self.data["ac@10"],
            "ac@20": self.data["ac@20"],
        })
        
