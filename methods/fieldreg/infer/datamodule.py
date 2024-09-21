
from typing import Tuple
from pathlib import Path

from methods.common.infer.base import InferDataModule
from methods.fieldreg.data.dataset import SampleDataset
from methods.fieldreg.data.transforms import ResizeOpenCV
from methods.fieldreg.data.playfield import Playfield

import cv2



class InferDataModule_Calib360(InferDataModule):
    def __init__(
        self,
        labels_filepath: Path,
        target_shape: Tuple[int,int] = (360, 640)
    ):
        self.folder = labels_filepath.parent
        self.target_shape = target_shape
        
        # Standard playfield
        self.playfield = Playfield(
            length=105,
            width=68,
            grid_shape=(8,7)
        )

        # Resize to specified resolution
        resize_transform = ResizeOpenCV(
            shape=target_shape,
            interpolation=cv2.INTER_AREA
        )

        # Configure our dataset        
        self.dataset = SampleDataset(
            root=self.folder.as_posix(),
            num_channels=15,
            load_images=True,
            load_sample=True,
            transform=resize_transform,
            transform_distance_map=resize_transform,
            grayscale=False
        )
        
    
    
    
    def get_inference_dataset(self):
        return self.dataset
    
    def get_playfield(self) -> Playfield:
        return self.playfield
