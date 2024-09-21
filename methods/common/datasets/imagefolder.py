
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        folder: str,
        search_pattern: str = "*.png"
    ):
        self.folder = Path(folder)
        self.files = sorted(list(self.folder.rglob(search_pattern)))
        
    def __len__(self):
        return len(self.files)
    
    
    def __getitem__(self, idx):
        image_filepath = self.files[idx].as_posix()
        
        image = self._load_image(image_filepath)
        homography = np.eye(3)
        
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
        image = cv2.resize(image, (1280,720), interpolation=cv2.INTER_AREA)
        return image