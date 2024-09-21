
from project.base.logging import Logger
import torch
import torch.nn as nn
import numpy as np

import methods.common.data.utils as utils


class ImagePreviewLogger(Logger):
    def __init__(
        self,
        experiment,
        num_rows: int=3
    ):
        self.experiment = experiment
        if (experiment is not None):
            self.tracker = experiment.tracker
        self.num_rows = num_rows

        # Will be sampled later
        self.samples = None
        self.images = None
        

    def on_training_start(self):
        
        datamodule = self.experiment.datamodule
        module = self.experiment.module

        # Get the images
        self.samples, images = self.sample_images(datamodule, module)
        if (len(images) > 0):
            self.images = torch.concatenate(
                images, 
                dim=0
            ).to(module.device)


    def on_epoch_end(self, epoch, stats):
        self.make_preview()


    def make_preview(self):

        # Get the model
        model = self.experiment.module.model
        if (isinstance(model, nn.DataParallel)):
            model = model.module

        # Get the preview results
        model.eval()
        items = self.process_images(model, self.images)
        model.train()

        # Arange into grid, and send to tracker
        log_images = {}
        # Unpack the list of dicts
        for item in items:
            for key,image in item.items():
                if (not key in log_images):
                    log_images[key] = []
                log_images[key].append(image)

        # Arange images into grids
        result = {}
        for key, images in log_images.items():
            result[key] = utils.make_grid(images, nrow=self.num_rows)

        # Send to tracker
        self.tracker.write_images(result)


    def sample_dataset(self, dataset, num_images):
        idx = np.random.choice(
            len(dataset), 
            size=(num_images,), 
            replace=False
        )
        samples = [ dataset[i] for i in idx ]
        return samples




    def sample_images(self, datamodule, module):
        """ Sample our datasets """
        return [], []
    

    def process_images(self, model, images):
        """ Returns list of dict[key,image] items """
        return []