
import torch
import numpy as np

from methods.common.data.utils import yards

from methods.common.loggers.homography_previewer import HomographyPreviewerLogger
from methods.fieldreg.data.playfield import Playfield




class KpsfrPreviewLogger(HomographyPreviewerLogger):
    def __init__(
        self,
        experiment,
        num_images: int = 4,
        homography_units: str = "yards"
    ):
        super().__init__(experiment, num_rows=3)
        self.num_images = num_images
        
        # Draw playing field image (grayscale)
        playfield_shape = (1080, 1920)
        self.pf = Playfield(grid_shape=(8,7))
        self.image_playfield = np.max(
            self.pf.draw_playing_field(shape=playfield_shape),
            axis=2
        )

        # Prepare scale matrix - Evil WC2014 homography is in yards !
        if (homography_units == "yards"):
            self.scale = np.array([
                [ yards(self.pf.length) / playfield_shape[1], 0.0, 0.0],
                [ 0, yards(self.pf.width) / playfield_shape[0], 0.0],
                [ 0, 0, 1]
            ])
        else:
            self.scale = np.eye(3)


    def sample_images(self, datamodule, module):
        """ Sample our datasets """

        # Return just a few images from training and validation
        samples = []
        samples += self.sample_dataset(datamodule.dataset_train(), self.num_images)
        samples += self.sample_dataset(datamodule.dataset_val(), self.num_images)

        # Select just the images
        images = [ s["rgb"].unsqueeze(0) for s in samples ]
        return samples, images


    def process_images(self, model, images):
        """ Returns list of dict[key,image] items """

        # Make previews
        result = []
        for i in range(images.shape[0]):
            
            # Get the predicted heatmaps
            with torch.no_grad():
                heatmap_logits = model("inference",
                        images[i:i+1,1:2],
                        lookup=self.samples[i]["lookup_gt"][0].to(images.device)
                    )

            # Get our image and heatmap
            image = self.to_image(images[i,1].detach().cpu())
            homo_gt = self.samples[i]["homography"][1]

            # Get the predicted homography
            homo, image_keypoints, heatmap = self.calib.find_homography(
                heatmap_logits[:,0,0]
            )

            # Draw predicted keypoints
            image = self.draw_keypoints(image, image_keypoints, color=(255,0,0))
            
            # Ground-truth visualization
            if (homo_gt is not None):
                try:
                    inv_homo_gt = np.linalg.inv(homo_gt) @ self.scale                    
                    image = self.draw_playfield(
                        image, 
                        self.image_playfield, 
                        inv_homo_gt,
                        color=(0,0,255), 
                        alpha=0.75,
                        flip=self.samples[i]["flip"][1]
                    )
                except:
                    # Homography might not be invertible
                    pass

            # Draw predicted playing field
            if (homo is not None):
                try:
                    inv_homo = np.linalg.inv(homo) @ self.scale
                    image = self.draw_playfield(image, self.image_playfield, inv_homo,
                        color=(255,0,0), alpha=1.0,
                        flip=False
                    )                                        
                except:
                    # Homography might not be invertible
                    pass

            # Store result
            result.append({
                "preview": image,
                "heatmap": heatmap
            })


        return result

