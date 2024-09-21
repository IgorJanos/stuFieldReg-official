
from typing import Any, Dict
from torch.utils.data import Dataset

import numpy as np




class InferDataModule:
    def __init__(self):
        pass


    def get_inference_dataset(self) -> Dataset:
        """ Return the dataset to run inference on """
        pass
    
    

class InferModule:
    def __init__(self):
        pass
    
    
    def setup(
        self,
        datamodule: InferDataModule
    ):
        """ Initialize inference proces with the given datamodule """
        pass
        
   
    def predict(
        self, 
        x: Any
    ) -> Dict:
        """ Predict the calibration information for the given dataset sample """
        return None
    
    
    
    