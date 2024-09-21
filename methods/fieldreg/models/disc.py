


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm


def leaky_relu():
    return nn.LeakyReLU(0.2)

def relu():
    return nn.ReLU()

def silu():
    return nn.SiLU()

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def normLayer(ch):
    return nn.Identity() #nn.InstanceNorm2d(ch)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])
    


class SkipLayerExcitation(nn.Module):
    def __init__(self, chin, chout):
        super().__init__()
        self.main = nn.Sequential(  
            nn.AdaptiveAvgPool2d(4), 
            conv2d(chin, chout, 4, 1, 0, bias=False), silu(),
            conv2d(chout, chout, 1, 1, 0, bias=False), nn.Sigmoid() 
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)
    

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            normLayer(out_planes), 
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)
    
class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1),
            normLayer(out_planes), 
            nn.LeakyReLU(0.2),
            conv2d(out_planes, out_planes, 3, 1, 1),
            normLayer(out_planes), 
            nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0),
            normLayer(out_planes), 
            nn.LeakyReLU(0.2)
            )

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2

class FastDiscriminator(nn.Module):
    def __init__(self, chin=(3+15), channels=[32, 64, 64, 128, 128, 128]):
        super().__init__()

        self.down_128 = nn.Sequential( 
                    conv2d(chin, channels[0], 5, 2, 2),
                    nn.LeakyReLU(0.2, inplace=True) 
                )
        
        self.down_64 = DownBlock(channels[0], channels[1])   
        self.down_32 = DownBlock(channels[1], channels[2])   
        self.down_16 = DownBlock(channels[2], channels[3])   
        self.down_8 = DownBlock(channels[3], channels[4])    
        self.down_4 = DownBlock(channels[4], channels[5])    
        self.final = conv2d(channels[5], 1, 4, 1, 0)

        self.se_128_16 = SkipLayerExcitation(channels[0], channels[3])
        self.se_64_8 = SkipLayerExcitation(channels[1], channels[4])
        self.se_32_4 = SkipLayerExcitation(channels[2], channels[5])

        self.dec = SimpleDecoder(channels[5], chin)
        

    def forward(self, imgs, real: bool):
        img_256 = F.interpolate(imgs, size=(256,256), mode="bilinear")

        f_128 = self.down_128(imgs)
        f_64 = self.down_64(f_128)
        f_32 = self.down_32(f_64)

        f_16 = self.se_128_16(f_128, self.down_16(f_32))
        f_8 = self.se_64_8(f_64, self.down_8(f_16))
        f_4 = self.se_32_4(f_32, self.down_4(f_8))

        # patchgan result
        result = self.final(f_4)

        # for real images try to reconstruct
        if (real):
            img_128 = F.interpolate(imgs, size=(128,128), mode="bilinear")
            rec_128 = self.dec(f_8)

            # break into chunks
            img_128 = torch.cat(torch.chunk(img_128, 6, dim=1), dim=0)
            rec_128 = torch.cat(torch.chunk(rec_128, 6, dim=1), dim=0)
            return result, img_128, rec_128

        return result





class SimpleDecoder(nn.Module):
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                normLayer(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)