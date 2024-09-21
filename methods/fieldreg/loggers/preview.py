
import torch
import numpy as np
import cv2

from methods.common.data.utils import yards
from methods.fieldreg.data.utils import resize_homo
from methods.common.loggers.homography_previewer import HomographyPreviewerLogger
from methods.fieldreg.data.playfield import Playfield
from methods.fieldreg.data.transforms import (
    DmapsToImage
)
from methods.fieldreg.data.calib import Calibrator
from project.data.transforms import TensorToNumpy
from torchvision.transforms import Compose
import skimage.segmentation as ss


class FieldRegPreviewLogger(HomographyPreviewerLogger):
    def __init__(
        self,
        experiment,
        num_images: int = 4,
        threshold: float = 0.9,
        playfield_shape = (8,7)
    ):
        super().__init__(experiment, num_rows=3)
        self.num_images = num_images

        # Draw playing field image (grayscale)
        playfield_image_shape = (1080, 1920)
        self.pf = Playfield(grid_shape=playfield_shape)
        self.image_playfield = np.max(
            self.pf.draw_playing_field(shape=playfield_image_shape),
            axis=2
        )

        self.dmaps_to_image = Compose([
            DmapsToImage(),
            TensorToNumpy()
        ])        
        self.calib = Calibrator(self.pf, threshold=threshold)
        self.scale = np.array([
            [ self.pf.length / playfield_image_shape[1], 0.0, 0.0],
            [ 0, self.pf.width / playfield_image_shape[0], 0.0],
            [ 0, 0, 1]
        ])
        self.scale_yards = np.array([
            [ yards(self.pf.length) / playfield_image_shape[1], 0.0, 0.0],
            [ 0, yards(self.pf.width) / playfield_image_shape[0], 0.0],
            [ 0, 0, 1]
        ])


    def sample_images(self, datamodule, module):
        """ Sample our datasets """

        # Return just a few images from training and validation
        samples = []
        samples += self.sample_dataset(datamodule.dataset_train(), self.num_images)
        samples += self.sample_dataset(datamodule.dataset_val(), self.num_images)

        # Select just the images
        images = [ s["image"].unsqueeze(0) for s in samples ]
        return samples, images

    def process_images(self, model, images):
        """ Returns list of dict[key,image] items """

        # Get the predictions
        with torch.no_grad():
            dmaps_hat = model(images)

            # Make previews
        result = []
        for i in range(images.shape[0]):           

            # Get our image and heatmap
            image = self.to_image(self.samples[i]["image"].clone().detach().cpu())
            dmaps = self.dmaps_to_image(dmaps_hat[i].detach().cpu())
            flip = self.samples[i]["flip"]
            homo_gt = self.samples[i]["homography"]

            # Get the predicted homography
            homo, image_keypoints = self.calib.find_homography(
                target_shape=(image.shape[0], image.shape[1]),
                dmaps=dmaps_hat[i]         
            )
            
            # Rescale to 720p
            image_720, homo = resize_homo(image, homo, target_shape=(720, 1280))
            _, homo_gt = resize_homo(image, homo_gt, target_shape=(720, 1280))
            

            # Draw predicted keypoints
            if (image_keypoints is not None):
                image_keypoints = ss.expand_labels(image_keypoints, distance=2)
                image_720 = self.draw_keypoints(image_720, image_keypoints, color=(255,0,0))

            # Ground-truth visualization
            if (homo_gt is not None):
                try:
                    scale_units = (self.scale_yards if self.samples[i]["units"] == "yards" else self.scale)                    
                    inv_homo = np.linalg.inv(homo_gt) @ scale_units                    
                    image_720 = self.draw_playfield(
                        image_720, self.image_playfield, inv_homo,
                        color=(0,0,255), alpha=0.75,
                        flip=flip
                    )
                except:
                    # Homography might not be invertible
                    pass

            # Draw predicted playing field
            if (homo is not None):
                try:
                    inv_homo = np.linalg.inv(homo) @ self.scale
                    image_720 = self.draw_playfield(image_720, self.image_playfield, inv_homo,
                        color=(255,0,0), alpha=1.0,
                        flip=False
                    )
                except:
                    # Homography might not be invertible
                    pass

            # Store result
            result.append({
                "preview": image_720,
                "distance_map": dmaps
            })

        return result

