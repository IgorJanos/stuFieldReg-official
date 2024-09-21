
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

import project as p


class RobustModule(p.SingleModelModule):
    def __init__(self, experiment):
        super().__init__(experiment)
        
        # Losses
        class_weights = 100.0 * torch.ones(
                experiment.cfg.common.num_classes,
                device=self.device
            )
        class_weights[0] = 1.0      # Background is less important
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        
        
    def train_iteration(self, epoch, iteration, batch, stats):       
        # Load inputs
        image = batch["image"].to(self.device)
        heatmap = batch["dilated_heatmap"].to(self.device).long()
        
        # Forward pass
        with autocast():
            heatmap_logits = self.model(image)
            loss = self.loss(heatmap_logits, heatmap)
        
        # Backward pass & update parameters
        self.opt.zero_grad()

        # Backward with autoscaler
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        # Update statistics
        stats.set("lr", self.scheduler.get_last_lr()[0])
        stats.step("loss", loss.item())       
        
        # Update scheduler
        super().train_iteration(epoch, iteration, batch, stats)
        
        
    
    def validate_iteration(self, epoch, iteration, batch, stats):
        # Load inputs
        image = batch["image"].to(self.device)
        heatmap = batch["dilated_heatmap"].to(self.device).long()
        
        # Forward pass
        heatmap_logits = self.model(image)
        loss = self.loss(heatmap_logits, heatmap)

        # Update statistics
        stats.step("val_loss", loss.item())       
        
        

