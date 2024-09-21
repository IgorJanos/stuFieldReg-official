
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvtf
import timm
import numpy as np
from functools import partial
from segmentation_models_pytorch.base import modules as md


def autocrop(x, shapeFrom):
    _,_,H,W = shapeFrom.shape
    y = tvtf.crop(x, 0, 0, H, W)
    return y

def join(x1, x2):
    return torch.cat([autocrop(x1,x2), x2], dim=1)


def upscale(x):
    return tnf.interpolate(
        x, scale_factor=2, mode="bilinear", 
        align_corners=False
    )

class SingleConvBlock(nn.Module):
    def __init__(self, chin, chout, k, s, p):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(chin, chout, k, s, p),
            nn.BatchNorm2d(chout),
            nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)

class DoubleConvBlock(nn.Module):
    def __init__(self, chin, chout, k, s, p):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(chin, chout, k, s, p),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
            nn.Conv2d(chout, chout, k, s, p),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class UpBlock(nn.Module):
    def __init__(self, chin, chskip, chout, convblock):
        super().__init__()
        self.chin = chin
        self.chskip = chskip
        self.chout = chout
        self.conv = convblock(chin=chin+chskip, chout=chout)

    def forward(self, x, xskip):
        if (xskip is not None):
            x = join(upscale(x), xskip)
        else:
            x = upscale(x)
        y = self.conv(x)
        return y
    

class CrossChannelSkipFC(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        num_features = shape[0] * shape[1]
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )

    def forward(self, x):
        B,C,H,W = x.shape
        y = x.view(B*C,-1).contiguous()
        y = self.mlp(y)
        y = y.view(B,C,H,W).contiguous()
        return x + y


class SEBlock(nn.Module):
    def __init__(self, chin1, chin2):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(chin1, chin2, 4, 1, 0),
            nn.BatchNorm2d(chin2),
            nn.ReLU(),
            nn.Conv2d(chin2, chin2, 1, 1, 0),
            nn.BatchNorm2d(chin2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        return self.main(x1) * x2


class NewModel(nn.Module):
    def __init__(
        self,
        encoder_name,
        chin=3,
        chout=18,
        input_shape=[384,640],
        dec_channels=[ 256, 256, 128, 128, 64 ]
    ):
        super().__init__()
        self.chout = chout

        # Pretrained encoder
        self.enc = timm.create_model(encoder_name, features_only=True, pretrained=True)
        self.ch_enc = [ fi["num_chs"] for fi in self.enc.feature_info.info ]

        # Shape for every level
        IH, IW = input_shape
        self.shapes = [
            (
                int(np.ceil(IH / (2**(1+i)))),
                int(np.ceil(IW / (2**(1+i)))),
            )
            for i in range(len(self.ch_enc))
        ]

        # Build or decoder
        #block_class = DoubleConvBlock
        block_class = SingleConvBlock
        convFN = partial(block_class, k=3, s=1, p=1)

        ch_enc = list(reversed(self.ch_enc))
        #ch_skip = ch_enc[1:] + [0]
        ch_skip = [ch_enc[1], ch_enc[2], ch_enc[3], 0, 0]
        self.ch_dec = dec_channels

        self.dec = nn.ModuleList()
        chin = ch_enc[0]
        for chskip,chout in zip(ch_skip, self.ch_dec):
            self.dec.append(
                UpBlock(chin, chskip, chout, convFN)
            )
            chin=chout

        chlast = self.ch_dec[-1]
        self.head = nn.Sequential(
            nn.Conv2d(chlast, self.chout, 1, 1, 0)
        )



    def forward(self, x):
        f = self.enc(x)
        y = f[-1]
        skip = list(reversed(f))[1:] + [None]

        u0 = self.dec[0](y, skip[0])
        u1 = self.dec[1](u0, skip[1])
        u2 = self.dec[2](u1, skip[2])
        u3 = self.dec[3](u2, None)  #skip[3])
        u4 = self.dec[4](u3, None)

        y = self.head(u4)

        #y = y.tanh()*0.5 + 0.5
        return y


    def get_encoder_params(self):
        return self.enc.parameters()



