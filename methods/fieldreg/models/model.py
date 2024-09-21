
import torch
from typing import List, Optional
import torch.nn as nn

import numpy as np

import torchvision.models as m
import torchvision.models.segmentation as s


from methods.fieldreg.models.unet import Unet

class SomeCrazyModel(nn.Module):
    def __init__(self, chin, chout):
        super().__init__()
        self.seg = s.deeplabv3_resnet50(
            weights=None,
            num_classes=chout,
            weights_backbone=m.ResNet50_Weights.IMAGENET1K_V1
        )


    def forward(self, x):
        result = self.seg(x)
        return result["out"]
    


class CenterBlock(nn.Module):
    def __init__(
            self,
            shape
        ):
        super().__init__()

        H,W = shape
        ch = H*W
        
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0),
        )



    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B,C,(H*W)).transpose(2,1).unsqueeze(3)     # B, (H*W), C, 1
        x = self.conv(x)
        x = x[:,:,:,0].transpose(2,1).view(B,C,H,W)
        return x



class SmpModel(nn.Module):
    def __init__(
            self, 
            chin, 
            chout,
            encoder_name: str = "resnet18",
            decoder_channels: List[int] = [256, 128, 64, 32, 16],
            input_shape: List[int] = [ 384, 640 ],
            num_channels_mlp: int = 16,
            decoder_attention_type: Optional[str] = None
        ):
        super().__init__()
        self.encoder_name = encoder_name
        self.model = Unet(
            encoder_name=encoder_name,      
            encoder_weights="imagenet",     
            in_channels=chin,               
            classes=chout,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type
        )

        # Now make a connected center block
        #IH, IW = input_shape
        #CH = int(np.ceil( IH / (2 ** (len(decoder_channels))) ))
        #CW = int(np.ceil( IW / (2 ** (len(decoder_channels))) ))
        #num_channels = self.model.encoder.out_channels[-1]
        #self.model.decoder.center = CenterBlock(shape=(CH,CW))


    def forward(self, x):
        result = self.model(x)
        # into <0;1>
        result = nn.functional.tanh(result) * 0.5 + 0.5
        return result
    

    def get_train_params(self):
        result = []
        result += list(self.model.decoder.parameters())
        return result

