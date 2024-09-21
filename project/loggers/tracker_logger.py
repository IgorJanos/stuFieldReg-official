
from project.base.logging import Logger
from project.base.tracker import Tracker



class TrackerLogger(Logger):
    def __init__(
        self,
        tracker: Tracker 
    ):
        self.tracker = tracker

    def on_epoch_end(self, epoch, stats):

        # Insert also epoch        
        epoch_stats = {
            "epoch": int(epoch)
        }
        epoch_stats.update(stats)
        
        self.tracker.write_scalars(epoch_stats)


class TrackerCommitLogger(Logger):        
    def __init__(
        self,
        tracker: Tracker 
    ):
        self.tracker = tracker
    
    def on_epoch_end(self, epoch, stats):
        self.tracker.commit()