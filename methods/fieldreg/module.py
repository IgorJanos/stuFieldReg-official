
import torch.nn as nn
from torch.cuda.amp import autocast

import project as p

from methods.fieldreg.losses import VGGLoss



class FieldRegModule(p.SingleModelModule):
    def __init__(self, experiment):
        super().__init__(experiment)
        
         # Losses
        self.loss_pixel = nn.MSELoss()
        #self.loss_perceptual = self.accelerate(VGGLoss())
        
        # Lambdas
        self.lambda_pixel = experiment.cfg.common.lambda_pixel
        #self.lambda_perceptual = experiment.cfg.common.lambda_perceptual

        
    def train_iteration(self, epoch, iteration, batch, stats):        
        # Load inputs
        image = batch["image"].to(self.device)
        dmap = batch["dmaps"].to(self.device)
        
        # Forward pass
        with autocast():
            dmap_hat = self.model(image)        
            loss_pixel = self.loss_pixel(dmap_hat, dmap)

        #loss_perceptual = self.lambda_perceptual*self.loss_perceptual(dmap_hat, dmap).mean()
        
        # Total loss
        loss = loss_pixel #+ loss_perceptual
        
        # Backward pass & update parameters
        self.opt.zero_grad()

        # Backward with autoscaler
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()
        
        # Update statistics
        stats.set("lr", self.scheduler.get_last_lr()[0])
        stats.step("loss", loss.item())
        #stats.step("lPixel", loss_pixel.item())
        #stats.step("lPerc", loss_perceptual.item())
        
        # Update scheduler
        super().train_iteration(epoch, iteration, batch, stats)


      
    
    def validate_iteration(self, epoch, iteration, batch, stats):
        image = batch["image"].to(self.device)
        dmap = batch["dmaps"].to(self.device)
        dmap_hat = self.model(image)

        # Forward pass
        dmap_hat = self.model(image)        
        loss_pixel = self.lambda_pixel*self.loss_pixel(dmap_hat, dmap)
        #loss_perceptual = self.lambda_perceptual*self.loss_perceptual(dmap_hat, dmap).mean()
        
        # Total loss
        loss = loss_pixel #+ loss_perceptual
                
        # Update statistics
        stats.step("val_loss", loss.item())
        #stats.step("val_lPixel", loss_pixel.item())
        #stats.step("val_lPerc", loss_perceptual.item())

