
from hydra.utils import instantiate
import project as p
from torch.utils.data import Dataset, DataLoader


class KpsfrDataModule(p.DataModule):
    def __init__(self, experiment):
        super().__init__(experiment)
        
        self.ds_train = instantiate(self.experiment.cfg.dataset.train)
        self.ds_val = instantiate(self.experiment.cfg.dataset.val)
        
        
    def setup(self):
        # Initialize loaders    
        self.loader_train = DataLoader(
            self.ds_train,
            batch_size=self.experiment.cfg.common.batch_size,
            num_workers=self.experiment.cfg.common.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        self.loader_val = DataLoader(
            self.ds_val,
            batch_size=self.experiment.cfg.common.batch_size,
            num_workers=self.experiment.cfg.common.num_workers,
            shuffle=False,
            pin_memory=True,
            #drop_last=True
        )

        
    def dataset_train(self) -> Dataset:
        return self.ds_train
    
    def dataset_val(self) -> Dataset:
        return self.ds_val
    
    def dataloader_train(self) -> DataLoader:
        return self.loader_train
    
    def dataloader_val(self) -> DataLoader:
        return self.loader_val