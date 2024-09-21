
from typing import Any, Dict
from pathlib import Path
import numpy as np

from methods.common.infer.base import InferModule, InferDataModule
from methods.fieldreg.loggers.preview import FieldRegPreviewLogger
from methods.fieldreg.data.utils import resize_homo
from methods.common.data.utils import yards

import project as p
import torch
import torch.nn as nn


class FieldRegInferModule(InferModule):
    def __init__(
        self,
        experiments_path: Path,
        experiment_name: str,
        
        make_images: bool=False
    ):
        # Get experiment folder
        experiment_path = experiments_path / experiment_name
        checkpoint_path = experiment_path / "checkpoints" / "last.pt"
        
        # Load from checkpoint
        self.experiment = p.Experiment.from_checkpoint(
            checkpoint_path
        )
        
        self.make_images = make_images
        
        # We use the logger to draw visualizations
        self.previewer = FieldRegPreviewLogger(
            self.experiment, num_images=1,
            playfield_shape=(14,7)
        )

    def setup(self, datamodule: InferDataModule):
        pass
    
    
    def predict(self, x: Any) -> Dict:
        """
            x - sample from dataset (including label)
        """
        
        # Prepare the model
        device = self.experiment.module.device
        model = self.experiment.module.model
        if (isinstance(model, nn.DataParallel)):
            model = model.module            
        model.eval()
        
        # Compute the prediction
        image = x["image"].to(device)        
        with torch.no_grad():
            dmaps = model(image.unsqueeze(0))

        # Rescale to 720p
        image = self.previewer.to_image(image.clone().detach().cpu())
            
        # Get the predicted homography
        homo, _ = self.previewer.calib.find_homography(
            target_shape=(image.shape[0], image.shape[1]),
            dmaps=dmaps[0]
        )
        
        if ("image_original" in x):
            
            image_original = self.previewer.to_image(x["image_original"])
            _, homo = resize_homo(
                image, homo, 
                target_shape=(image_original.shape[0], image_original.shape[1])
            )
            image_720p = image_original            
            
        else:            
            image_720p, homo = resize_homo(image, homo, target_shape=(720, 1280))
        
        # Draw predicted playing field
        if (homo is not None):            
            # to yards
            to_yards = np.array([
                [ yards(1.0), 0, 0 ],
                [ 0, yards(1.0), 0 ],
                [ 0, 0, 1]
            ])
            homo = to_yards @ homo
                        
            try:
                inv_homo = np.linalg.inv(homo) @ self.previewer.scale_yards
                image_720p = self.previewer.draw_playfield(
                    image_720p, 
                    self.previewer.image_playfield, 
                    inv_homo,
                    color=(255,0,0), alpha=1.0,
                    flip=False
                )
            except:
                # Homography might not be invertible
                pass        
        
        # Extract homography matrix
        result = {
            "homography": homo,
            #"image_720p": image_720p
        }
        
        if (self.make_images):
            result["image_720p"] = image_720p
        
        return result        


