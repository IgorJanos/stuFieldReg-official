
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from pathlib import Path

from project.base.module import Module
from project.base.datamodule import DataModule
from project.base.trainer import Trainer
from project.base.tracker import Tracker, TrackerCompose

from project.loggers.tracker_logger import TrackerLogger, TrackerCommitLogger
from project.trackers.wandb_tracker import WandBTracker
from project.loggers.checkpointer import ModelCheckpoint
from project.trackers.filesystem_tracker import FileSystemTracker, HierarchicalFormatter

from hydra.utils import instantiate

import torch
import logging


BASE_PATH = Path(".scratch/experiments")


class Experiment:
    def __init__(
        self,
        cfg: DictConfig,
        load_checkpoint_filepath: Optional[Path] = None,
        base_path: Path = BASE_PATH
    ):
        self.cfg = cfg
        self.base_path = base_path
        self.experiment_name = f"{str(cfg.exp.name)}/{str(cfg.exp.ver)}"
        self.experiment_path = base_path / self.experiment_name
        
        # Extract from config
        self.start_epoch = 0
        self.max_epochs = cfg.common.max_epochs
        
        # Load checkpoint ?
        if (load_checkpoint_filepath is not None):
            self.setup()
            self.load_checkpoint(load_checkpoint_filepath)
        
        
    @staticmethod
    def from_checkpoint(checkpoint_filepath: Path):
        
        experiment_path = checkpoint_filepath.parent.parent        
        cfg_filepath = experiment_path / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(cfg_filepath)
        
        experiment = Experiment(cfg, load_checkpoint_filepath=checkpoint_filepath)       
        return experiment
        
        
    def setup(self):
        """ Create and prepares internal objects """        
        self.datamodule = self.create_datamodule()
        self.module = self.create_module()
        self.tracker = self.create_tracker()
        
                        
        
    def train(self):
                                                
        """ Execute new training """        
        # Create the trainer object
        self.trainer = self.create_trainer(
            datamodule=self.datamodule,
            module=self.module,
            start_epoch=self.start_epoch,
            max_epochs=self.max_epochs
        )
        self.datamodule.setup()
        self.tracker.setup()
        
        # Setup logging
        logs=[
            TrackerLogger(self.tracker),
            ModelCheckpoint(
                self, 
                singleFile=False, 
                interval=self.cfg.common.checkpoint_interval,
                epoch_interval=self.cfg.common.checkpoint_epoch_interval
            )
        ]
        
        # Extra loggers
        if (self.cfg.loggers is not None):
            for logger_config in self.cfg.loggers:
                logs.append(instantiate(logger_config, experiment=self))
                
        # Final tracker commit
        logs.append(TrackerCommitLogger(self.tracker))
        
        self.trainer.setup(logs)
        
        # Commence training
        self.trainer.fit()
        
        # Shutdown tracker
        self.tracker.close()
    
    
    def create_tracker(self) -> Tracker:
        """ Creates experiment tracker object """
        
        trackers = [
            FileSystemTracker(self, filepath_formatter=HierarchicalFormatter())
        ]
        
         # Extra loggers
        if (self.cfg.trackers is not None):
            for tracker_config in self.cfg.trackers:
                
                tracker = instantiate(tracker_config, experiment=self)
                
                # Skip wandb ?
                skip = False
                if (isinstance(tracker, WandBTracker) and (not self.cfg.exp.wandb)):
                    skip = True
                
                if (not skip):
                    trackers.append(tracker)
        
        return TrackerCompose(trackers)
    
    
    def create_module(self) -> Module:
        """ Creates module for training """        
        # Create module based on current configuration
        return instantiate(self.cfg.module, experiment=self)

    
    def create_datamodule(self) -> DataModule:
        
        """ Creates datamodule for training """
        # Create module based on current configuration
        return instantiate(self.cfg.datamodule, experiment=self)
    
    
    def create_trainer(
        self,
        datamodule: DataModule,
        module: Module,
        start_epoch: int,
        max_epochs: int
    ) -> Trainer:
        """ Creates trainer object """

        # Default implementation
        return Trainer(
            datamodule=datamodule,
            module=module,
            start_epoch=start_epoch,
            max_epochs=max_epochs
        )        
    
    def save_checkpoint(self, checkpoint_filename):
        
        checkpoint = {
            "module": self.module.to_checkpoint()
        }
        
        # Save to file
        file_path = self.experiment_path / "checkpoints" / checkpoint_filename
        logging.info(f"Saving checkpoint: {file_path.as_posix()}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, file_path.as_posix())
        
    
    def load_checkpoint(self, checkpoint_filepath):
        
        checkpoint = torch.load(
            checkpoint_filepath.as_posix(),
            map_location=torch.device("cpu")
        )
        
        # Load module from checkpoint
        self.module.load_checkpoint(checkpoint["module"])



