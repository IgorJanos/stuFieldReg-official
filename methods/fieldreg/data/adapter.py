
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

import methods.common.data.augmentation as aug

from methods.robust.data.utils import gen_im_partial_grid
from methods.fieldreg.data.utils import resize_homo


from methods.common.data.utils import yards
from methods.fieldreg.data.augmentation import RandomHorizontalFlip
from methods.fieldreg.data.conversion import HomographyToDistanceMap
from methods.fieldreg.data.playfield import Playfield
from methods.fieldreg.eval.camera import Camera

from project.data.transforms import NumpyToTensor


class FieldRegDatasetAdapter(Dataset):
    """
        Adapter class - wraps around raw dataset object, and
        converts ground truth homography matrix into the 
        distance maps.
        
        result = {           
            "image":        - network input tensor
            "homography":   - 3x3 homography matrix
            "dmaps":        - distance maps
        }
    """
    def __init__(
        self, 
        dataset: Dataset,
        noise_translate: float=0.0,
        noise_rotate: float=0.0,
        random_flip: bool=False,
        playfield_shape = (8,7),
        max_count = -1,
        target_shape = (360, 640),
        keep_original: bool= False
    ):
        self.dataset = dataset
        self.playfield_shape = playfield_shape
        self.target_shape = target_shape
        self.keep_original = keep_original

        # Decide if augmentation is to be used
        if (noise_translate == 0.0 and noise_rotate == 0.0):
            self.augment = aug.WarpAugmentation(
                warp_function=gen_im_partial_grid,
                mode="test",
                noise_translate=0.0,
                noise_rotate=0.0
            )
        else:
            self.augment = aug.WarpAugmentation(
                warp_function=gen_im_partial_grid,
                mode="train",
                noise_translate=noise_translate,
                noise_rotate=noise_rotate
            )

        # Flip augmentation
        self.flip = RandomHorizontalFlip(playfield_shape, random_flip)
        self.image_transform = transforms.Compose([
            NumpyToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.dmap_transform = NumpyToTensor()

        self.conversion = HomographyToDistanceMap(
            playfield_grid_shape=playfield_shape,
            distance_map_radius=4,
            distance_map_shape=(1920, 1080) #(2560, 1440)
        )

        self.count = len(self.dataset)
        if (max_count > 0):
            self.count = min(self.count, max_count)


    def __len__(self):
        return self.count


    def __getitem__(self, idx):
        
        # Read from inner dataset
        sample = self.dataset[idx]

        # Augment
        image_original, grid, homography = self.augment(
            image=sample["image"], 
            homography=sample["homography"],
            frame_idx=0
        )
        image = image_original

        # Compute distance maps
        inv_homo = np.linalg.inv(homography)
        if (sample["units"] == "yards"):
            inv_homo = inv_homo @ self.conversion.scale

        dmaps = self.conversion.to_distance_map(
            inv_homo,
            output_shape=(image.shape[0], image.shape[1])
        )

        # Flip image and dmaps
        image, dmaps, flip = self.flip(image, dmaps)

        # Resize
        img_360, homo = resize_homo(image, homography, target_shape=self.target_shape)
        dmaps_360 = cv2.resize(dmaps, (self.target_shape[1], self.target_shape[0]), interpolation=cv2.INTER_LINEAR)

        # Transform & yield
        result = {
            "image": self.image_transform(img_360.copy()),
            "homography": homo,
            "units": sample["units"],
            "dmaps": self.dmap_transform(dmaps_360.copy()),
            "flip": flip
        }
        
        if (self.keep_original):
            result["image_original"] = self.image_transform(image_original)
        
        return result
    
    
    def _resize(self, x, shape):
        H,W = shape
        if ((x.shape[0] == H) and (x.shape[1] == W)):
            return x
        return cv2.resize(x, (W,H), interpolation=cv2.INTER_LINEAR)
    
    

class Calib360Adapter(Dataset):
    def __init__(
        self, 
        dataset: Dataset,
        target_shape = (360, 640)
    ):
        self.dataset = dataset
        self.target_shape = target_shape

        # We need this just for the dimensions of playing field
        pf = Playfield()
        self.offset = np.array([
            [1, 0, pf.length/2.0 ],
            [0, 1, pf.width/2.0 ],
            [0, 0, 1]
        ])
        


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        
        # Read from inner dataset
        sample = self.dataset[idx]

        # Resize input images to 720p
        src_image = sample["image"]
        if ((src_image.shape[0] == self.target_shape[0]) and 
            (src_image.shape[1] == self.target_shape[1])
            ):
            image = src_image
        else:
            image = cv2.resize(
                src_image, 
                dsize=(self.target_shape[1], self.target_shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )

        # Convert homography to yards... grr
        homo_meters = self._make_homography(
                image=image,
                sample=sample["sample"]
            )
        to_yards = np.array([
            [ yards(1), 0, 0 ],
            [ 0, yards(1), 0 ],
            [ 0, 0, 1]
        ])
        
        homo_yards = to_yards @ homo_meters
        
        # Sanity check!
        assert homo_yards.shape[0] == 3

        result = {
            # Compute homography matrix for this image
            "image": image,
            "homography": homo_yards,            
            "sample": sample["sample"],
            "units": "yards"
        }

        return result



    def _make_homography(self, image, sample):
        cam = Camera()
        cam.from_calib360_sample(
            sample, 
            width=image.shape[1],
            height=image.shape[0]
        )
        h = cam.get_homography()
        h = self.offset @ h
        return h
    
    

    
    
    