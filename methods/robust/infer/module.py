import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Dict
from methods.common.infer.base import *
from methods.common.infer.base import InferDataModule
import project as p

from methods.robust.loggers.preview import RobustPreviewLogger
from methods.common.data.utils import yards



class RobustInferModule(InferModule):
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
        self.previewer = RobustPreviewLogger(
            self.experiment, num_images=1
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
            heatmap = model(image.unsqueeze(0))
            
        # Get the predicted homography
        homo, _, _ = self.previewer.calib.find_homography(heatmap[0])
        
        # Rescale to 720p
        image_720p = self.previewer.to_image(image.clone().detach().cpu())
        
         # Draw predicted playing field
        if (homo is not None):            
            # to yards
            to_yards = np.array([
                [ yards(1.0), 0, 0 ],
                [ 0, yards(1.0), 0 ],
                [ 0, 0, 1]
            ])
            #homo = to_yards @ homo
                        
            try:
                inv_homo = np.linalg.inv(homo) @ self.previewer.scale
                image_720p = self.previewer.draw_playfield(
                    image_720p, 
                    self.previewer.image_playfield, 
                    inv_homo,
                    color=(255,0,0), alpha=1.0,
                    flip=False
                )
            except:
                # Homography might
                pass
            
        # Extract homography matrix
        result = {
            "homography": homo
        }
        
        if (self.make_images):
            result["image_720p"] = image_720p
        
        return result