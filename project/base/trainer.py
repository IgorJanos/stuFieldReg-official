
    
from project.base.datamodule import DataModule
from project.base.module import Module
from project.base.logging import LogCompose
from project.base.utils import Statistics

import torch
from tqdm.auto import tqdm

    
    
class Trainer:
    def __init__(
        self,
        datamodule: DataModule,
        module: Module,
        start_epoch: int=0,
        max_epochs: int=10
    ):
        self.datamodule = datamodule
        self.module = module
        self.start_epoch = start_epoch
        self.max_epochs = max_epochs
        
        
    def setup(self, logs=[]):
        self.log = LogCompose(logs)
        
        
    def fit(self):
        
        # Training loop
        self.log.on_training_start()
        
        for epoch in range(self.start_epoch, self.max_epochs):
            
            # Training phase
            stats_train = Statistics()
            self.train_epoch(epoch, stats_train)
            
            # Validation phase
            stats_val = Statistics()
            self.validate_epoch(epoch, stats_val)
            
            # Log completion
            self.log.on_epoch_end(
                epoch,
                stats=Statistics.merge(stats_train, stats_val)
            )
                
        self.log.on_training_end()
        
    
    
    
    def train_epoch(self, epoch, stats):
        
        # Get loader if possible
        loader = self.datamodule.dataloader_train()
        if (loader is not None):                        
            self.module.train_epoch_started(epoch, stats, loader)
            
            # Iterate over data
            progress = tqdm(
                enumerate(loader),
                desc=f"Train {epoch}",
                total=len(loader),
                ascii=True
            )
            
            # Train all iterations
            for iteration, batch in progress:
                self.module.train_iteration(epoch, iteration, batch, stats)
                self.log.on_iteration(epoch, iteration)        
                progress.set_postfix(stats.get())
        
            # Done    
            self.module.train_epoch_finished(epoch, stats)
            
    
    
    def validate_epoch(self, epoch, stats):
        
        # Get loader if possible
        loader = self.datamodule.dataloader_val()
        if (loader is not None):
            self.module.validate_epoch_started(epoch, stats, loader)
            
            # Iterate over data
            progress = tqdm(
                enumerate(loader),
                desc=f"Val {epoch}",
                total=len(loader),
                ascii=True
            )
            
            # Train all iterations
            for iteration, batch in progress:
                with torch.no_grad():
                    self.module.validate_iteration(epoch, iteration, batch, stats)
                progress.set_postfix(stats.get())
            
            # Done    
            self.module.validate_epoch_finished(epoch, stats)
