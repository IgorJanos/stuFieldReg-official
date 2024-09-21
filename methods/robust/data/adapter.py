
import numpy as np
from torch.utils.data import Dataset

import skimage.segmentation as ss

from torchvision import transforms

from methods.common.data.utils import to_torch
import methods.robust.data.utils as utils

from project.data.transforms import NumpyToTensor
import methods.common.data.augmentation as aug


class RobustDatasetAdapter(Dataset):
    """
        Adapter class - wraps around raw dataset object, and
        converts ground truth homography matrix into the heatmap 
        of keypoints.
        
        result = {           
            "image":            - network input tensor
            "homography":       - 3x3 homography matrix
            "heatmap":          - keypoint heatmap
            "dilated_heatmap":  - dilated keypoint heatmap
        }
    
    """
    def __init__(
        self, 
        dataset: Dataset,
        noise_translate: float=0.0,
        noise_rotate: float=0.0,
        random_flip: bool=False,
        max_count = -1
    ):
        self.dataset = dataset
        
        # Decide if augmentation is to be used
        if (noise_translate == 0.0 and noise_rotate == 0.0):
            self.augment = aug.WarpAugmentation(
                warp_function=utils.gen_im_partial_grid,
                mode="test",
                noise_translate=0.0,
                noise_rotate=0.0
            )
        else:
            self.augment = aug.WarpAugmentation(
                warp_function=utils.gen_im_partial_grid,
                mode="train",
                noise_translate=noise_translate,
                noise_rotate=noise_rotate
            )

        # Flip augmentation
        self.flip = aug.LeftRightFlipAugmentation(enabled=random_flip)  
        self.image_transform = transforms.Compose([
            NumpyToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.count = len(self.dataset)
        if (max_count > 0):
            self.count = min(self.count, max_count)
        
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        
        # Read from inner dataset
        sample = self.dataset[idx]
        
        # Augment
        image, grid, homography = self.augment(
            image=sample["image"], 
            homography=sample["homography"],
            frame_idx=0
        )
        image, grid, is_flip = self.flip(image, grid)
        
        # Convert to heatmap & dilate
        heatmap = self._get_heatmap(image, grid)

        # Transform & yield
        result = {
            "image": self.image_transform(image.copy()),
            "homography": homography,
            "units": sample["units"],
            "heatmap": to_torch(heatmap),
            "dilated_heatmap": to_torch(ss.expand_labels(heatmap, distance=5)),
            "flip": is_flip
        }
       
        return result
    
    
    
    def _get_heatmap(self, image, grid):
        """ 
            Returns a <H;W> matrix, where each element is either 0 (background)
            or CLASS of the respective keypoint
        """
        
        # Smaller resolution
        height = image.shape[0] // 4
        width = image.shape[1] // 4
        
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Contains only visible keypoints
        num_pts = grid.shape[0]
        for idx in range(num_pts):
            px = np.rint(grid[idx, 0] / 4).astype(np.int32)
            py = np.rint(grid[idx, 1] / 4).astype(np.int32)
            if (0 <= px < width) and (0 <= py < height):
                heatmap[py, px] = grid[idx, 2]
        
        return heatmap