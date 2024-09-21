


from typing import Dict


class Logger:
    def __init__(self):
        pass

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_epoch_end(
        self, 
        epoch: int, 
        stats: Dict[str, float]
    ):
        pass

    def on_iteration(self, epoch, it):
        pass



class LogCompose(Logger):
    def __init__(self, loggers=[]):
        self.loggers = loggers

    def on_training_start(self):
        for logger in self.loggers:
            logger.on_training_start()

    def on_training_end(self):
        for logger in self.loggers:
            logger.on_training_end()

    def on_epoch_end(
        self, 
        epoch: int, 
        stats: Dict[str, float]
    ):
        for logger in self.loggers:
            logger.on_epoch_end(epoch, stats)

    def on_iteration(self, epoch, it):
        for logger in self.loggers:
            logger.on_iteration(epoch, it)



