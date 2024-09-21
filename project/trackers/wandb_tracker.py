
from project.base.tracker import Tracker

from omegaconf import OmegaConf
import wandb



class WandBTracker(Tracker):
    def __init__(
        self, 
        experiment, 
        project,
        entity,
        log_dir
    ):
        self.experiment = experiment
        self.run = None
        self.project = project
        self.entity = entity
        self.log_dir = log_dir
        
        # Accumulator
        self.acc = {}
        
        
    def setup(self):
        self.run = wandb.init(
            config = OmegaConf.to_container(self.experiment.cfg),
            project = self.project,
            entity = self.entity,
            name = self.experiment.experiment_name,
            reinit = True,
            dir = self.log_dir
        )

    
    def close(self):
        if self.run is not None:
            wandb.finish()
            self.run = None
        
        
    def write_scalars(self, items: dict):
        """ Writes scalar values """
        self.acc.update(items)
    
    def write_images(self, items: dict):
        """ Writes images """        
        result = {}
        for key, value in items.items():                        
            result[key] = wandb.Image(value)

        self.acc.update(result)
                       
    def write_figures(self, items: dict):
        """ Writes figures """
        self.acc.update(items)

    def commit(self):
        """ Push accumulated objects """
        if (self.acc is not None):
            self.run.log(self.acc)
            self.acc = {}
