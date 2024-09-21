
import numpy as np
from typing import Callable, Any
import methods.common.data.utils as utils
import random



class WarpAugmentation(Callable):
    def __init__(
        self,
        warp_function: Any,
        mode: str="train",
        noise_translate: float=0.0,
        noise_rotate: float=0.0
    ):
        # Template
        self.template_grid = utils.gen_template_grid()
        self.warp_fn = warp_function
        self.mode = mode
        self.noise_translate = noise_translate
        self.noise_rotate = noise_rotate

    def __call__(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        frame_idx: int      
    ):
        warp_image, warp_grid, warp_homography = self.warp_fn(
            mode=self.mode,
            frame=image,
            f_idx=frame_idx,
            gt_homo=homography,
            template=self.template_grid,
            noise_trans=self.noise_translate,
            noise_rotate=self.noise_rotate,
            index=-1        # not really used ...
        )
        return warp_image, warp_grid, warp_homography




class LeftRightFlipAugmentation(Callable):
    def __init__(self, enabled: bool=False):
        self.enabled = enabled
        
    def __call__(self, image, grid):
        if (self.enabled):            
            if (random.random() < 0.5):
                image, grid = utils.put_lrflip_augmentation(image, grid)
                return image, grid, True
            
        return image, grid, False
