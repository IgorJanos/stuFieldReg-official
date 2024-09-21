from typing import Any, Dict
from methods.common.infer.base import *



class LabelInferModule(InferModule):
    def __init__(self):
        pass        
    
    def setup(self, datamodule: InferDataModule):
        pass
    
    
    def predict(self, x: Any) -> Dict:
        """
            x - sample from dataset (including label)
        """
                    
        # Extract homography matrix
        result = {
            "homography": x["homography"]
        }
        
        return result