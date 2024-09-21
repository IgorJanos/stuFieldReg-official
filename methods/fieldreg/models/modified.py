from typing import List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock, DecoderBlock
from segmentation_models_pytorch.base import SegmentationHead
from functools import partial

from methods.fieldreg.models.non_local import NONLocalBlock2D


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class GatedConv(nn.Module):
    def __init__(self, chin, chout, k):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chin, chout*2, kernel_size=k, stride=1, padding=((k-1) // 2)),
            nn.BatchNorm2d(chout*2),
            GLU()
        )

    def forward(self, x):
        return self.conv(x)
    

class ConvBlock(nn.Module):
    def __init__(self, chin, chout, k):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chin, chout, k, stride=1, padding=((k-1) // 2)),
            nn.BatchNorm2d(chout),
            nn.LeakyReLU(0.02)
        )

    def forward(self, x):
        return self.conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_dim, k=8):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//k, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//k, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) 

    def forward(self,x):
        B,C,H,W = x.size()
        query = self.query_conv(x).view(B,-1,H*W).permute(0,2,1)   # B X CX(N)
        key = self.key_conv(x).view(B,-1,H*W)                      # B X C x (*W*H)
        energy = torch.bmm(query, key)                             # transpose check
        attention = self.softmax(energy)                           # BX (N) X (N) 
        value = self.value_conv(x).view(B,-1,H*W)                  # B X C X N

        out = torch.bmm(value,attention.permute(0,2,1) )
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out #, attention




class ModifiedConvBlock(nn.Module):
    def __init__(
        self,
        chin,
        chskip,
        chmiddle,
        chout,
        attention_type=None,
        nonlocal_x=False,
        nonlocal_skip=False,
        gated_kernel_size=1,
        use_selfattention=False
    ):
        super().__init__()

        self.nonlocal_x = (NONLocalBlock2D(chin, chin) if nonlocal_x else nn.Identity())
        self.nonlocal_skip = (NONLocalBlock2D(chskip, chskip) if nonlocal_skip else nn.Identity())

        self.conv1 = ConvBlock(
            chin+chskip+(2+chmiddle),              # 2 more for UV
            chout, 
            k=3
            )  

        if (gated_kernel_size > 0):
            self.conv2 = GatedConv(chout, chout, k=gated_kernel_size)      
        else:
            self.conv2 = ConvBlock(chout, chout, k=3)

        self.attention1 = md.Attention(attention_type, in_channels=chin+chskip)
        self.attention2 = (SelfAttention(chout) if use_selfattention else nn.Identity())

    def forward(self, x, skip=None, uv=None, fmiddle=None):
        x = self.nonlocal_x(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            skip = self.nonlocal_skip(skip)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        # Rescale UV for our desired size
        uv = F.interpolate(uv, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        fmiddle = F.interpolate(fmiddle, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        x = torch.cat([x, uv, fmiddle], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class ModifiedCenterBlock(nn.Module):
    def __init__(self, ch, shape, k=8):
        super().__init__()
        self.k = k
        H,W = shape
        num_features = H*W*(ch // k)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(),
        )


    def forward(self, x):
        # Break into chunks
        B,C,H,W = x.shape
        num_chunks = self.k
        new_channels = C // self.k
        x_chunks = x.view(B*num_chunks, new_channels*H*W)

        # Transform!
        y = self.mlp(x_chunks)
        y = y.view(B,C,H,W)
        return y

class ModifiedUnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        gated_kernel: List[int] = [3, 3, 0, 0, 0],
        chmiddle: int = 8
    ):
        super().__init__()
        self.grid_shape = None

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        H,W = 384,640
        MH, MW = (H // (2**n_blocks)), (W // (2**n_blocks))
        self.center = ModifiedCenterBlock(head_channels, shape=(MH,MW), k=32)

        # Middle projection
        self.middle = nn.Sequential(
            nn.Conv2d(head_channels, chmiddle, 1, 1, 0),
            nn.BatchNorm2d(chmiddle),
            nn.LeakyReLU()
        )

        block_fn = [
            partial(ModifiedConvBlock, gated_kernel_size=gated_kernel[0], chmiddle=chmiddle, use_selfattention=True),
            partial(ModifiedConvBlock, gated_kernel_size=gated_kernel[1], chmiddle=chmiddle),
            partial(ModifiedConvBlock, gated_kernel_size=gated_kernel[2], chmiddle=chmiddle),
            partial(ModifiedConvBlock, gated_kernel_size=gated_kernel[3], chmiddle=chmiddle),
            partial(ModifiedConvBlock, gated_kernel_size=gated_kernel[4], chmiddle=chmiddle),
        ]

        # combine decoder keyword arguments
        kwargs = dict(attention_type=attention_type)
        blocks = [
            block(chin=in_ch, chskip=skip_ch, chout=out_ch, **kwargs)
            for block, in_ch, skip_ch, out_ch in zip(block_fn, in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, features):


        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        # Get our Global Context stuff
        fmiddle = self.middle(x)
        uv = self._get_uv_maps(shape=(x.size(2), x.size(3)))
        uv = uv.clone().to(x.device).repeat(x.size(0), 1, 1, 1)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip, uv, fmiddle)

        return x

    def _get_uv_maps(self, shape):        
        if (self.grid_shape != shape):
            self.grid_shape = shape
            H,W = shape
            u = torch.arange(W) / (W-1.0)
            v = torch.arange(H) / (H-1.0)
            vv,uu = torch.meshgrid(v,u, indexing="ij")
            self.uv = torch.cat([ vv.unsqueeze(0), uu.unsqueeze(0) ], dim=0).unsqueeze(0)
        return self.uv




class ModifiedUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        middle_channels: int = 8,
        classes: int = 1,
        final_kernel_size: int = 1,
        activation: Optional[Union[str, callable]] = None,
        gated_kernel: List[int] = [3, 3, 0, 0, 0]
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = ModifiedUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            attention_type=decoder_attention_type,
            gated_kernel=gated_kernel,
            chmiddle=middle_channels
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=final_kernel_size,
        )

        self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(x, features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    






class ModifiedSegmentationModelUnet(nn.Module):
    def __init__(
        self,
        chin: int=3,
        chout: int=15,
        encoder_name: str="resnet18",
        encoder_depth: int=5,
        encoder_weights: str="imagenet",
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_attention_type: str="none",
        final_kernel_size: int=1,
        gated_kernel: List[int] = [3, 3, 0, 0, 0],
        middle_channels: int = 8
    ):
        super().__init__()

        # None!
        if encoder_name == "none": encoder_name = None
        if decoder_attention_type == "none": decoder_attention_type = None

        # Build the UNET model from segmentation models pytorch
        self.model = ModifiedUnet(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            in_channels=chin,
            classes=chout,
            activation=None,
            final_kernel_size=final_kernel_size,
            gated_kernel=gated_kernel,
            middle_channels=middle_channels
        )

    def forward(self, x):

        B,C,H,W = x.shape
        p = 0
        if (H < 384):
            p = (384-H) // 2
            x = F.pad(x, pad=(0,0,p,p), mode="constant", value=0)

        y = self.model(x)

        if (H < 384):
            # remove padding
            y = y[:,:,p:p+H,:]

        return y