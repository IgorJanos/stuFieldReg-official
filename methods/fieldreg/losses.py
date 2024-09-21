
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import models, transforms





class VGGLoss(nn.Module):
    MODELS = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=12, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.model = self.MODELS[model](
                weights=models.VGG16_Weights.IMAGENET1K_V1
            ).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(input)

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target):        
        # Reshape into 3-channel images
        input = self._reshape_into_rgb(input)
        target = self._reshape_into_rgb(target)
        
        # Return perceptual loss
        sep = input.shape[0]
        batch = torch.cat([input, target])
        feats = self.get_features(batch)
        input_feats, target_feats = feats[:sep], feats[sep:]
        
        l = F.mse_loss(input_feats, target_feats, reduction=self.reduction)
        return l



    def _reshape_into_rgb(self, x):
        num_chunks = x.size(1) // 3        
        return torch.cat(torch.chunk(x, num_chunks, dim=1), dim=0)