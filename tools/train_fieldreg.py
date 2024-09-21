import os
import hydra

from omegaconf import DictConfig, OmegaConf

import project as p
import logging



@hydra.main(version_base=None, config_path="config_fieldreg", config_name="default")
def do_main(cfg: DictConfig):
        
    logging.info("Training script - FieldReg started")
        
    experiment = p.Experiment(cfg)
    experiment.setup()
    experiment.train()

    logging.info("Training script - FieldReg finished.")
    


if __name__ == "__main__":
    do_main()
