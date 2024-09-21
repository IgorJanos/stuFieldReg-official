
import numpy as np
from typing import Callable, Any
import methods.common.data.utils as utils
import random



class RandomHorizontalFlip(Callable):
    def __init__(self, playfield_shape, enabled: bool=False):
        self.enabled = enabled
        self.hline_count = playfield_shape[0]
        
    def __call__(self, image, dmaps):
        is_flip = False
        if (self.enabled):            
            if (random.random() < 0.5):
                
                image = np.fliplr(image)
                dmaps = np.fliplr(dmaps)

                # Also change order of channels
                dmaps_out = dmaps.copy()

                # Indices of horizontal lines
                for i in range(self.hline_count):
                    dmaps_out[:,:,i] = dmaps[:,:,(self.hline_count-1)-i]

                dmaps = dmaps_out
                is_flip = True

        return image, dmaps, is_flip

