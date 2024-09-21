from typing import Any

from torch.utils.data import Dataset
from pathlib import Path

import pandas as pd
import numpy as np
import cv2

from methods.fieldreg.data.sample import Sample
from methods.fieldreg.data.calib import (
    load_calib360_image, load_calib360_dmaps,
    load_calib360_image_pair
)


import re
import ast


def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))




class SampleDataset(Dataset):
    def __init__(
        self, 
        root: str,
        num_channels: int = 15,
        load_images: bool = True,
        load_dmaps: bool = True,
        load_sample: bool = True,
        transform: Any = None,
        transform_distance_map: Any = None,
        grayscale: bool = False,
        scale_shape: Any = None
    ):
        self.root = Path(root)
        self.num_channels = num_channels
        self.grayscale = grayscale
        self.scale_shape = scale_shape
        self.load_images = load_images
        self.load_dmaps = load_dmaps
        self.load_sample = load_sample
        self.transform = transform
        self.transform_distance_map = transform_distance_map
        self.df = pd.read_csv((self.root / "labels.csv").as_posix())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):        
        row = self.df.iloc[index]
        sample = self._row_to_sample(row)

        result = {}

        # Do we need dmaps
        if (self.load_dmaps):
            distance_map = load_calib360_dmaps(
                (self.root / sample.image_name).as_posix(),
                num_channels=self.num_channels,
            )
            distance_map = self._resize(distance_map)            
            if self.transform_distance_map:
                distance_map = self.transform_distance_map(distance_map)

            result["distance_map"] = distance_map


        # Do we need images ?
        if (self.load_images):
            image = load_calib360_image(
                (self.root / sample.image_name).as_posix(),
                grayscale=self.grayscale
            )
            image = self._resize(image)
            if self.transform:
                image = self.transform(image)            

            result["image"] = image


        if (self.load_sample):
            result["sample"] = sample

        return result

    def get(self, index):
        row = self.df.iloc[index]
        sample = self._row_to_sample(row)

        result = {}

        if (self.load_images):
            image, distance_map = load_calib360_image_pair(
                (self.root / sample.image_name).as_posix(),
                num_channels=self.num_channels,
                grayscale=self.grayscale
            )
            
            image = self._resize(image)
            distance_map = self._resize(distance_map)
            
            result["image"] = image
            result["distance_map"] = distance_map

        result["sample"] = sample
        return result

    def _row_to_sample(self, row):
        result = Sample(
            image_name = row["name"],
            x = row["x"],
            y = row["y"],
            z = row["z"],
            pan = row["pan"],
            tilt = row["tilt"],
            roll = row["roll"],
            fov_v = row["fov_v"],
            fov_h = row["fov_h"]
        )
        return result
    
    def _resize(self, x):
        if (self.scale_shape is not None):
            return cv2.resize(x, (self.scale_shape[1], self.scale_shape[0]))
        return x
    

class PredictionsDataset(Dataset):
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):        
        row = self.df.iloc[index]   
        
        homo = row["homography"]
        try:
            homo = str2array(homo)
        except:
            homo = None
        
        result = {
            "item": row["item"],
            "homography": homo
        }
            
        return result


