
import numpy as np
import skimage.segmentation as ss
import cv2

from torchvision import transforms

from project.data.transforms import TensorToNumpy


from methods.common.loggers.image_preview import ImagePreviewLogger
from methods.common.data.transforms import UnNormalize
from methods.common.data.calib import Calibrator


class HomographyPreviewerLogger(ImagePreviewLogger):
    def __init__(
        self,
        experiment,
        num_rows: int=3,
    ):
        super().__init__(experiment, num_rows)

        self.calib = Calibrator()        
        self.to_image = transforms.Compose([
            UnNormalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            TensorToNumpy()
        ])




    def draw_keypoints(self, image, image_keypoints, color=(255,0,0)):
        """ Upscales keypoints map into image resolution and 
            overlays it over the image
        """

        # Get keypoints image in target image resolution
        a = ss.expand_labels(image_keypoints, distance=1)
        a = cv2.resize(
            a, 
            dsize=(image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        a = np.expand_dims(a, axis=2)

        # Alpha of keypoints image
        a = (a > 0)*1.0

        # Color of keypoints image
        c = np.concatenate([a*color[0], a*color[1], a*color[2]], axis=2)

        # Superimpose the keypoints
        result = (1.0-a)*image + a*c
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result


    def draw_playfield(
        self, 
        image, 
        image_playfield, 
        homography, 
        color=(255,0,0),
        alpha=1.0,
        flip=False
    ):
        """ Draws the playfield image under the homography matrix
            over the target image
        """
        if (homography is None):
            return image
        
        # Warp the playfield image
        warp_field = cv2.warpPerspective(
            image_playfield,
            homography,
            (image.shape[1], image.shape[0]),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0)
        )
        
        if (flip):
            warp_field = np.fliplr(warp_field)

        # Get the alpha
        a = np.expand_dims((warp_field / 255.0), axis=2)

        # Color of playfield
        c = np.concatenate([a*color[0], a*color[1], a*color[2]], axis=2)
        
        # Draw with specified alpha
        a = a * alpha

        # Superimpose the playing field image
        result = (1.0-a)*image + a*c
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

