import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Dict
from methods.common.infer.base import *
import project as p

from methods.kpsfr.loggers.preview import KpsfrPreviewLogger




class KpsfrInferModule(InferModule):
    def __init__(
        self,
        experiments_path: Path,
        experiment_name: str
    ):
        
        # Get experiment folder
        experiment_path = experiments_path / experiment_name
        checkpoint_path = experiment_path / "checkpoints" / "last.pt"
        
        # Load from checkpoint
        self.experiment = p.Experiment.from_checkpoint(
            checkpoint_path
        )
        
        # We use the logger to draw visualizations
        self.previewer = KpsfrPreviewLogger(
            self.experiment, num_images=1
        )
        
        
    def load_pretrain(self, filepath):
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        
        model = self.experiment.module.model
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to("cuda")
        
        self.experiment.module.model = model
        
    
    
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
        image = x["rgb"].to(device).unsqueeze(0)    
        lookup = x["lookup"].to(device)
        with torch.no_grad():
            heatmap = model("inference", 
                            image[:,0:1],
                            lookup[0]
                            )
            
        # Get the predicted homography
        homo, _, _ = self.previewer.calib.find_homography(
            heatmap[:,0,0]
        )
            
        # Extract homography matrix
        result = {
            "homography": homo
        }
        
        return result