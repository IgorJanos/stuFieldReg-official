
from typing import Any
from torch.utils.data import Dataset


class EvalDataModule:
    def __init__(self):
        pass
    
    def get_eval_dataset(self) -> Dataset:
        """ Return the dataset to run evaluation on """
        pass
    
    
    
    
class EvalModule:
    def __init__(self):
        pass
    
    
    def setup(
        self,
        data_gt: EvalDataModule,
        data_pred: EvalDataModule
    ):
        """ Initialize evaluation process with the given data modules """
        pass
    
    
    def evaluate(self, logger):
        """ Run the evaluation """
        pass
