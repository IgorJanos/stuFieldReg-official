from typing import List
import torch
import torch.nn as nn
import numpy as np

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead

class SegmentationModelUnet(nn.Module):
    def __init__(
        self,
        chin: int=3,
        chout: int=15,
        encoder_name: str="resnet18",
        encoder_depth: int=5,
        encoder_weights: str="imagenet",
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: str="none"
    ):
        super().__init__()

        # None!
        if encoder_name == "none": encoder_name = None
        if decoder_attention_type == "none": decoder_attention_type = None

        # Build the UNET model from segmentation models pytorch
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=chin,
            classes=chout,
            activation=None
        )

        self.model.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=chout,
            activation=None,
            kernel_size=1,
        )


    def forward(self, x):
        y = self.model(x)
        return y