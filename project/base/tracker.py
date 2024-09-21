
from typing import Dict
from pathlib import Path


class Tracker:
    def __init__(
        self,
        experiment
    ):
        self.experiment = experiment
              
              
    def setup(self):
        pass
    
    def close(self):
        pass
              
        
    def write_scalars(self, items: dict):
        """ Writes scalar values """
        pass
    
    
    def write_images(self, items: dict):
        """ Writes images """        
        pass    
    
    def write_figures(self, items: dict):
        """ Writes figures """
        pass
    
    def write_artifact(self, name) -> Path:
        """ Returns a full file path for the given artifact """
        return Path("")
    
    def commit(self):
        """ Process accumulated stuff """
        pass
    
    
class TrackerCompose:
    def __init__(
        self,
        trackers
    ):
        self.trackers = trackers
    
    
    def setup(self):
        for t in self.trackers:
            t.setup()
    
    def close(self):
        for t in self.trackers:
            t.close()
              
        
    def write_scalars(self, items: dict):
        for t in self.trackers:
            t.write_scalars(items)
    
    
    def write_images(self, items: dict):
        for t in self.trackers:
            t.write_images(items)
    
    def write_figures(self, items: dict):
        for t in self.trackers:
            t.write_figures(items)
    
    def write_artifact(self, name) -> Path:
        # Exception - we only forward to the first tracker
        if (len(self.trackers) > 0):
            return self.trackers[0].write_artifact(name)
        return Path("")
    
    def commit(self):
        for t in self.trackers:
            t.commit()
