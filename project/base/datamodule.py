
from omegaconf import DictConfig

from torch.utils.data import Dataset, DataLoader
       
    
    
class DataModule:
    def __init__(self, experiment):
        self.experiment = experiment
    
    def setup(self):
        """ Activate dataloaders """
        pass
    
    def dataset_train(self) -> Dataset:
        """ Access to training dataset """
        return None
    
    def dataset_val(self) -> Dataset:
        """ Access to validation dataset """
        return None
    
    def dataloader_train(self) -> DataLoader:
        """ Access to training dataloader """
        return None
    
    def dataloader_val(self) -> DataLoader:
        """ Access to validation dataloader """
        return None
    