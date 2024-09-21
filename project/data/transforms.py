
from typing import Callable
import torch
import numpy as np



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


class TensorToNumpy(Callable):
    """ Converts CHW Float (0,1) -> HWC UINT8 (0,255) """
    def __call__(self, x):
        x = x.cpu().detach().numpy()
        x = np.transpose(x, axes=(1,2,0))
        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return x