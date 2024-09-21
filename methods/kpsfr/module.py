
import torch
import torch.nn as nn
import torch.nn.functional as F

import project as p
from methods.kpsfr.losses.binary_dice import BinaryDiceLoss

from torch.cuda.amp import autocast



class KpsfrModule(p.SingleModelModule):
    def __init__(self, experiment):
        super().__init__(experiment)
        
        self.num_objects = experiment.cfg.common.num_objects
        
        # Losses
        class_weights = 100.0 * torch.ones(self.num_objects + 1, device=self.device)
        class_weights[0] = 1.0      # Background is less important        
        self.wce = nn.CrossEntropyLoss(weight=class_weights)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss()
        
        # Load pretrained checkpoint
        pretrain_checkpoint_path = experiment.cfg.common.pretrain_checkpoint
        if (pretrain_checkpoint_path is not None):
            checkpoint = torch.load(
                pretrain_checkpoint_path, 
                map_location=torch.device("cpu")
            )        
            model = self.model
            if (isinstance(model, nn.DataParallel)):
                model = model.module
            model.load_state_dict(checkpoint["model_state_dict"])        
            self.model = self.accelerate(model)           
            # Recreate the optimizer
            self.opt = self.create_optimizer(self.model)
            self.scheduler = self.create_scheduler(self.opt)
        
        # Don't use scaler
        self.use_scaler = False    

        
        
    def train_iteration(self, epoch, iteration, batch, stats):       
        # Load inputs
        image = batch["rgb"].to(self.device)
        heatmap = batch["target_dilated_hm"].to(self.device)
        cls = batch["cls_gt"].to(self.device).long()
        selector = batch["selector"].to(self.device)
        lookup = batch["lookup"].to(self.device)
                
        if (self.use_scaler):                
            # Get our loss
            with autocast():
                loss = self._compute_iteration(image, heatmap, cls, selector, lookup)
                                
            # Backward pass & update parameters
            self.opt.zero_grad()
            # Backward with autoscaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss = self._compute_iteration(image, heatmap, cls, selector, lookup)                               
            # Backward pass & update parameters
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
                    
        
        # Update statistics
        stats.set("lr", self.scheduler.get_last_lr()[0])

        if (torch.isnan(loss).sum() == 0):
            stats.step("loss", loss.item())       
        
        # Update scheduler
        super().train_iteration(epoch, iteration, batch, stats)
        
        
    
    def validate_iteration(self, epoch, iteration, batch, stats):
        # Load inputs
        image = batch["rgb"].to(self.device)
        heatmap = batch["target_dilated_hm"].to(self.device)
        cls = batch["cls_gt"].to(self.device).long()
        selector = batch["selector"].to(self.device)
        lookup = batch["lookup"].to(self.device)
        
        # Get our loss
        loss = self._compute_iteration(image, heatmap, cls, selector, lookup)

        # Update statistics
        stats.step("val_loss", loss.item())       





    def _compute_iteration(self, image, heatmap, cls, selector, lookup):
       
        # Forward pass
        kf32, kf16, kf8, kf4 = self.model("encode_key", image)        
       
        # TODO: random pick 4
        ref_hm = heatmap.clone()
        ref_v = []
        for idx in range(self.num_objects):
            chunks = torch.split(ref_hm, [1, self.num_objects - 1], dim=1)
            mask = chunks[0]  # b*1*t*h*w
            other_masks = chunks[1]  # b*(objs-1)*t*h*w
            fg_mask = torch.zeros_like(mask)
            # TODO: Check label in the previous heatmap appears in the current heatmap or not
            for b in range(lookup.shape[0]):
                if lookup[b, 0, idx] not in lookup[b, 1].tolist():  # non-overlap
                    pass
                    # print('set to zero map')
                else:
                    fg_mask[b, 0, 0] = mask[b, 0, 0]

            out_v = self.model(
                'encode_value', 
                image[:, 0], 
                kf32[:, 0], 
                fg_mask[:, :, 0], 
                isFirst=True
            )
            ref_v.append(out_v)
            ref_hm = torch.cat([other_masks, mask], dim=1)

        ref_v = torch.stack(ref_v, dim=1)  # b*k*c*t*h*w

        # Segment qframe 1(k32[:, :, 1]) with mframe 0(k32[:, :, 0:1])
        prev_x, prev_logits, prev_heatmap = self.model(
            'segment', 
            kf32[:, 1], kf16[:, 1], kf8[:, 1], kf4[:, 1], 
            self.num_objects, 
            lookup[:, 1], selector[:, 1]
        )


        # TODO: random pick 4
        prev_hm = prev_heatmap.clone().detach()
        prev_v = []
        for idx in range(self.num_objects):
            chunks = torch.split(prev_hm, [1, self.num_objects - 1], dim=1)
            mask = chunks[0]  # b*1*h*w
            other_masks = chunks[1]  # b*(objs-1)*h*w

            fg_mask = torch.zeros_like(mask)

            # TODO: Check label in the previous heatmap appears in the current heatmap or not
            for b in range(lookup.shape[0]):
                if lookup[b, 1, idx] not in lookup[b, 2].tolist():  # non-overlap
                    pass
                    # print('set to zero map')
                else:
                    fg_mask[b, 0] = mask[b, 0]
            out_v = self.model('encode_value', image[:, 1], kf32[:, 1], fg_mask, isFirst=False)
            prev_v.append(out_v)
            prev_hm = torch.cat([other_masks, mask], dim=1)

        prev_v = torch.stack(prev_v, dim=1)  # b*k*c*t*h*w


        # Segment qframe 2(k32[:, :, 2]) with mframe 0 and 1(k32[:, :, 0:2])
        this_x, this_logits, this_heatmap = self.model(
            'segment', 
            kf32[:, 2], kf16[:, 2], kf8[:, 2], kf4[:, 2], 
            self.num_objects, 
            lookup[:, 2], selector[:, 2]
        )


        total_loss = 0.0
        b = heatmap.shape[0]
        size = heatmap.shape[-2:]

        prev_x = F.interpolate(prev_x, size, mode='bilinear', align_corners=False)
        prev_logits = F.interpolate(prev_logits, size, mode='bilinear', align_corners=False)
        prev_heatmap = F.interpolate(prev_heatmap, size, mode='bilinear', align_corners=False)

        this_x = F.interpolate(this_x, size, mode='bilinear', align_corners=False)
        this_logits = F.interpolate(this_logits, size, mode='bilinear', align_corners=False)
        this_heatmap = F.interpolate(this_heatmap, size, mode='bilinear', align_corners=False)


        for i in range(1, self.num_objects + 1):
            for j in range(b):
                loss_1 = 0.0
                loss_2 = 0.0
                if selector[j, 1, i-1] != 0:
                    loss_1 = self.dice(prev_heatmap[j:j+1, i-1:i], heatmap[j:j+1, i-1:i, 1]) + \
                        self.bce(prev_x[j:j+1, i-1:i], heatmap[j:j+1, i-1:i, 1]) + \
                        self.wce(prev_logits[j:j+1], cls[j:j+1, 1])

                if selector[j, 2, i-1] != 0:
                    loss_2 = self.dice(this_heatmap[j:j+1, i-1:i], heatmap[j:j+1, i-1:i, 2]) + \
                        self.bce(this_x[j:j+1, i-1:i], heatmap[j:j+1, i-1:i, 2]) + \
                        self.wce(this_logits[j:j+1], cls[j:j+1, 2])

                total_loss += loss_1 + loss_2
        total_loss = total_loss / (self.num_objects * 2.) / b / 4        
        
        # Return the computed loss
        return total_loss
        
        

