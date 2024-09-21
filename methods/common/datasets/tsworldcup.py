

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
import cv2


def get_clip_from_filepath(file_path):
    return file_path.parts[-3]+"/"+file_path.parts[-2]

def _is_descendant(base_path, file_path, clips):
    clip_name = get_clip_from_filepath(file_path)
    if (clip_name in clips):
        return True
    return False

def _read_np_homographyMatrix(filepath):
    return np.load(filepath)
    
    
       

class TsWorldCupDataset(Dataset):
    def __init__(
        self,
        subset_filepath: str
    ):
        subset_filepath = Path(subset_filepath)
        self.folder = subset_filepath.parent        
        self.clips = subset_filepath.read_text().splitlines()
        
        # Filter by clip
        dataset_folder = self.folder / "Dataset"
        self.files = []
        all_files = sorted(list(dataset_folder.rglob("*.jpg")))
        for fp in all_files:
            if (_is_descendant(self.folder, fp, self.clips)):
                self.files.append(fp.relative_to(self.folder))
        

    def __len__(self):
        return len(self.files)    
    
    def __getitem__(self, idx):
        image_filepath = self.files[idx].as_posix()
        clip = get_clip_from_filepath(self.files[idx])
        homography_filepath = image_filepath.replace("Dataset/", "Annotations/")
        homography_filepath = homography_filepath.replace(".jpg", "_homography.npy")        
    
        image = self._load_image(self.folder / image_filepath)
        homography = _read_np_homographyMatrix(self.folder / homography_filepath)
                               
        result = {
            "image": image,
            "homography": homography,
            "clip": clip,
            "units": "yards"
        }
        return result
      
    
    
    def _load_image(self, filepath):
        image = cv2.imread(filepath.as_posix(), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image




