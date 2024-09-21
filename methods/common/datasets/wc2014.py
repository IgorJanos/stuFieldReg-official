

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2

    
    
    
def _read_txt_homographyMatrix(filepath):
    return np.loadtxt(filepath)



class WorldCup2014Dataset(Dataset):
    def __init__(
        self,
        folder: str  
    ):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.rglob("*.jpg")))
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        image_filepath = self.files[idx].as_posix()
        homography_filepath = image_filepath.replace(".jpg", ".homographyMatrix")
        
        image = self._load_image(image_filepath)
        homography = _read_txt_homographyMatrix(homography_filepath)
        
        result = {
            "image": image,
            "homography": homography,
            "clip": "undefined",
            "units": "yards"
        }
        return result
    
    def _load_image(self, filepath):
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
   
