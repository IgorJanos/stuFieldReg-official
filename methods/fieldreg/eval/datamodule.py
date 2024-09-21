
from pathlib import Path
from torch.utils.data import Dataset

from methods.fieldreg.data.dataset import PredictionsDataset
from methods.fieldreg.eval.base import EvalDataModule
from methods.fieldreg.data.playfield import Playfield



class EvalDataModule_Calib360(EvalDataModule):
    def __init__(
        self,
        predictions_filepath: Path
    ):
        # Standard playfield
        self.playfield = Playfield(
            length=105,
            width=68,
            grid_shape=(14,7)
        )
                
        # Configure our dataset
        self.dataset = PredictionsDataset(
            filepath=predictions_filepath
        )
        
    def get_eval_dataset(self) -> Dataset:
        """ Return the dataset to run evaluation on """
        return self.dataset