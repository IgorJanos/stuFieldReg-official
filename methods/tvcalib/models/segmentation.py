
from typing import Union
from pathlib import Path
import torch

from torchvision.models.segmentation import deeplabv3_resnet101
from SoccerNet.Evaluation.utils_calibration import SoccerPitch



class InferenceSegmentationModel:
    def __init__(self, checkpoint: Union[str, Path], device) -> None:
        self.device = device
        self.model = deeplabv3_resnet101(
            num_classes=len(SoccerPitch.lines_classes) + 1, aux_loss=True
        )
        self.model.load_state_dict(torch.load(checkpoint)["model"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def inference(self, img_batch):
        return self.model(img_batch)["out"].argmax(1)
