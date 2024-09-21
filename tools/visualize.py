import argparse
import cv2
from pathlib import Path
import pandas as pd
from functools import partial

from methods.common.infer.base import InferDataModule, InferModule

# Datasets
from methods.robust.data.adapter import RobustDatasetAdapter
from methods.kpsfr.data.adapter import KpsfrDatasetAdapter
from methods.fieldreg.data.adapter import Calib360Adapter, FieldRegDatasetAdapter
from methods.fieldreg.data.dataset import SampleDataset
from methods.common.datasets.wc2014 import WorldCup2014Dataset
from methods.robust.loggers.preview import RobustPreviewLogger

from infer import DataModule
from tqdm import tqdm
import project as p
import numpy as np


import re
import ast


def str2array(s):
    # Remove space after [
    s=re.sub('\[ +', '[', s.strip())
    # Replace commas and spaces
    s=re.sub('[,\s]+', ', ', s)
    return np.array(ast.literal_eval(s))


def create_datamodule(method, scenario) -> InferDataModule:    
    dataset = None    
    adapter_class = RobustDatasetAdapter
    
    # Check method first
    if (method == "our"):
        adapter_class = partial(
            FieldRegDatasetAdapter,
            playfield_shape=(14,7),
            target_shape=(720, 1280)
        )
    elif (method == "chu"):
        adapter_class = KpsfrDatasetAdapter
    
    # Create data ...
    if ((scenario == "wc") or (scenario == "mix")):
        dataset = adapter_class(
            dataset=WorldCup2014Dataset(
                folder=".scratch/dataset/wc2014/soccer_data/test"
            ),
            noise_rotate=0,
            noise_translate=0,
            random_flip=False
        )        
    elif (scenario == "calib360"):        
        shape = (720, 1280)        
        dataset = adapter_class(
            dataset=Calib360Adapter(
                dataset=SampleDataset(
                    root=".scratch/data/calib360-val", #/mnt/datasets/fiit/calib360/val",
                    load_images=True,
                    load_dmaps=False,
                    load_sample=True,
                    grayscale=False
                ),
                target_shape=shape
            ),
            noise_rotate=0,
            noise_translate=0,
            random_flip=False            
        )            
    return DataModule(dataset=dataset)


def write_image(folder: Path, idx, image):
    filepath = folder / f"{idx:05d}.png"
    folder.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath.as_posix(), image)


def main(args):
    
    # Split inference file into method and scenario
    method, scenario = args.inference_file.stem.split("-")
    datamodule = create_datamodule(method, scenario)
    out_path = args.output_dir / f"{method}-{scenario}"
    
    # Load inference data
    data = pd.read_csv(args.inference_file)
    
    # Helper class to visualize stuff
    logger = RobustPreviewLogger(None, 1)
        
    # Visualize
    with tqdm(
        datamodule.get_inference_dataset(),
        ascii=True,
        ncols=80
        ) as progress:        
        for idx, x in enumerate(progress):
            if (idx >= len(data)):
                break
            row = data.iloc[idx]
            
            image = logger.to_image(x["image"])
            homo = row["homography"]
            try:
                homo = str2array(homo)
            except:
                homo = None
            
            if (homo is not None):
                try:
                    inv_homo = np.linalg.inv(homo) @ logger.scale
                    image = logger.draw_playfield(
                        image,
                        logger.image_playfield,
                        inv_homo,
                        color=(255,0,0),
                        flip=False
                    )
                except:
                    # Homography might not be invertible
                    pass
            
            # Write
            write_image(out_path, idx, image)
     



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    
    p.add_argument("--inference_file", "-i",
                   default=".scratch/inference/chu-wc.csv",
                   type=Path,
                   help="Inference CSV file")
    
    p.add_argument("--output_dir", "-od", 
                   default=".scratch/inference/images",
                   type=Path,
                   help="Output dir to store images into")
    
    main(p.parse_args())