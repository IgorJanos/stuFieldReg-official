
from typing import Callable

import torch
import numpy as np
from pathlib import Path
import random
import cv2

from pytorch_histogram_matching import Histogram_Matching


class ResizeOpenCV(Callable):
    def __init__(self, shape, interpolation=cv2.INTER_AREA):
        self.shape = (shape[1], shape[0])       # OpenCV - width, height
        self.interpolation = interpolation
        
    def __call__(self, x):
        x = cv2.resize(x, self.shape, interpolation=self.interpolation)
        return x


class DmapsToImage(Callable):
    def __call__(self, x):
        ch = x.size(-3)
        num_chunks = (ch + 2) // 3
        x_chunks = torch.chunk(x, num_chunks, dim=-3)
        y = torch.stack(x_chunks).sum(dim=0)
        return y


class NumpyToTensor(Callable):
    """ Converts HWC UINT8 (0,255) into CHW FLOAT (0,1) """    
    def __call__(self, x):
        x = np.transpose(x, (2,0,1))
        if (x.dtype == np.uint8):
            x = torch.from_numpy(x)
            x = (x / 255.0).float()
        else:
            x = torch.from_numpy(x).float()
        return x

class GaussianNoise(Callable):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class ColorOffset(Callable):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        C,H,W = x.shape
        offset = torch.randn(size=(C,1,1)) * self.sigma
        return x + offset


class NullTransform(Callable):
    def __call__(self, x):
        return x
    


TO_TORCH=NumpyToTensor()


class HistogramMatching(Callable):
    def __init__(self, p, root: str):
        self.p = p
        self.root = Path(root)
        self.source_files = list(self.root.rglob("*.jpg"))
        self.hm = Histogram_Matching(differentiable=False)
        
    def __call__(self, x):

        # Select source image
        if (random.random() < self.p):
            src_filepath = random.choice(self.source_files)
            src_image = cv2.imread(src_filepath.as_posix(), cv2.IMREAD_COLOR)
            src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
            src_image = TO_TORCH(src_image)

            # Match our histogram
            matched = self.hm(x.unsqueeze(0), src_image.unsqueeze(0))[0]
            return matched

        # No histogram matching
        return x