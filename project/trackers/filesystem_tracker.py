
from project.base.tracker import Tracker
from project.base.utils import NamedCounter
from project.loggers.csv_logger import CSVWriter
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.figure as fig

import cv2


class HierarchicalFormatter(Callable):
    def __init__(self):
        pass
    
    def __call__(self, idx):
        return f"{idx//1000:04d}XXX/{idx:07d}"




class FileSystemTracker(Tracker):
    def __init__(
        self,
        experiment,
        filepath_formatter=HierarchicalFormatter()
    ):
        self.experiment = experiment
        self.counter_images = NamedCounter()
        self.counter_figures = NamedCounter()
        self.filepath_formatter = filepath_formatter
        self.stats_writer = CSVWriter(
            self.experiment.experiment_path / "training.csv"
        )
        
        
    def setup(self):
        self.stats_writer.open()

    
    def close(self):
        self.stats_writer.close()
        
        
    def write_scalars(self, items: dict):
        """ Writes scalar values """
        self.stats_writer.write(items)
    
    
    def write_images(self, items: dict):
        """ Writes images """        
        for key, value in items.items():            
            idx = self.counter_images.step(key)
            filepath = self._get_local_filepath(
                "images",
                f"{key}/{self.filepath_formatter(idx)}"
            )            
            self._write_image(filepath, value)
            

    
    def write_figures(self, items: dict):
        """ Writes figures """
        for key, value in items.items():            
            idx = self.counter_figures.step(key)
            filepath = self._get_local_filepath(
                "figures",
                f"{key}/{self.filepath_formatter(idx)}"
            )            
            self._write_figure(filepath, value)
    
    
    
    def write_artifact(self, name) -> Path:
        """ Returns a full file path for the given artifact """
        return self._get_local_filepath("artifacts", name)



    def _write_image(self, filepath, image):        
        # Convert and save
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(filepath.as_posix() + ".jpg", image)


    def _write_figure(self, filepath, figure: fig.Figure):
        # Save as image
        figure.savefig(
            filepath.as_posix() + ".png"
        )
    

    def _get_local_filepath(self, kind, name) -> Path:
        
        # Plain and simple
        result = self.experiment.experiment_path / kind / name
        
        return result