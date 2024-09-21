
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
from typing import Any

import torch.optim.lr_scheduler as scheduler
from torch.cuda.amp import GradScaler

    
class Module:
    def __init__(self, experiment):
        self.experiment = experiment
    
    
    def to_checkpoint(self) -> dict:
        """ Save all important internals into state dictionary """
        return {}
    
    def load_checkpoint(self, state: dict):
        """ Load all important internals from state dictionary """
        pass
    
    
    def train_epoch_started(self, epoch, stats, dataloader):
        """ Called when an epoch is about to be started """
        pass
    
    def train_epoch_finished(self, epoch, stats):
        """ Called when an epoch training is finished """
        pass
    
    def train_iteration(self, epoch, iteration, batch, stats):
        """ Called for each training iteration """
        pass
    
    def validate_epoch_started(self, epoch, stats, dataloader):
        """ Called when an epoch is about to be validated """
        pass
    
    def validate_iteration(self, epoch, iteration, batch, stats):
        """ Called for each validation iteration """
        pass

    def validate_epoch_finished(self, epoch, stats):
        """ Called when an epoch validation is finished """
        pass





def decide_device():
    if (torch.cuda.is_available()): return "cuda"
    #if (torch.backends.mps.is_available()): return "mps"
    return "cpu"



class SingleModelModule(Module):
    def __init__(self, experiment):
        super().__init__(experiment)
        
        self.device = torch.device(decide_device())
        
        # Create model and optimizer
        self.model = self.accelerate(self.create_model())
        self.opt = self.create_optimizer(self.model)
        self.scheduler = self.create_scheduler(self.opt)

        # Autoscaler
        self.scaler = GradScaler(
            init_scale=32768,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000
        )



    def create_model(self) -> torch.nn.Module:        
        """ Creates an instance of our only trainable model """
        return instantiate(self.experiment.cfg.model)
        
    def create_optimizer(self, model) -> torch.optim.Optimizer:
        """ Create a single optimizer for our trainable model """
        return instantiate(self.experiment.cfg.optimizer, params=model.parameters())

    def create_scheduler(self, opt) -> Any:    
        """ Create scheduler object for the given optimizer """
        return instantiate(self.experiment.cfg.scheduler, optimizer=opt)
     
    def accelerate(self, model):        
        if (torch.cuda.is_available()):  
            model = model.to(self.device)      
            if (torch.cuda.device_count() > 1):
                return torch.nn.DataParallel(model)            
        return model

        
         
    
    def train_epoch_started(self, epoch, stats, dataloader):
        self.model.train()

    def train_iteration(self, epoch, iteration, batch, stats):
        # Update scheduler
        if (
            isinstance(self.scheduler, scheduler.CosineAnnealingWarmRestarts) or
            isinstance(self.scheduler, scheduler.CosineAnnealingLR)
            ):
            self.scheduler.step()
        else:
            pass


    def train_epoch_finished(self, epoch, stats):
        # Update scheduler as necessary
        if (
            isinstance(self.scheduler, scheduler.CosineAnnealingWarmRestarts) or
            isinstance(self.scheduler, scheduler.CosineAnnealingLR)
            ):
            pass
        else:
            # Only once per epoch
            self.scheduler.step()
        


    def validate_epoch_started(self, epoch, stats, dataloader):
        self.model.eval()

    
    def to_checkpoint(self) -> dict:
        """ Save all important internals into state dictionary """
        
        if (isinstance(self.model, torch.nn.DataParallel)):
            model = self.model.module
        else:
            model = self.model
            
        checkpoint = {
            "model": model.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict()
        }
            
        return checkpoint

    def load_checkpoint(self, state: dict):
        """ Load all important internals from state dictionary """
        if (isinstance(self.model, torch.nn.DataParallel)):
            model = self.model.module
        else:
            model = self.model
            
        model.load_state_dict(state["model"])
        self.model = self.accelerate(model)
        # Recreate the optimizer
        self.opt = self.create_optimizer(self.model)
        self.opt.load_state_dict(state["opt"])
        # Also the scaler
        if ("scaler" in state):
            self.scaler.load_state_dict(state["scaler"])


