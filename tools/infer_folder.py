import argparse
import cv2
from functools import partial
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from pathlib import Path

from methods.common.infer.base import InferDataModule, InferModule
from methods.common.infer.sink import PredictionsSink

from methods.common.infer.module import LabelInferModule
from methods.robust.infer.module import RobustInferModule
from methods.kpsfr.infer.module import KpsfrInferModule
from methods.fieldreg.infer.module import FieldRegInferModule
from methods.tvcalib.infer.module import TvCalibInferModule


# Datasets
from methods.robust.data.adapter import RobustDatasetAdapter
from methods.kpsfr.data.adapter import KpsfrDatasetAdapter
from methods.fieldreg.data.adapter import Calib360Adapter, FieldRegDatasetAdapter
from methods.fieldreg.data.dataset import SampleDataset
from methods.common.datasets.wc2014 import WorldCup2014Dataset
from methods.common.datasets.imagefolder import ImageFolderDataset


BASE_PATH = Path(".scratch/experiments")

EXPERIMENTS = {
    "nie": {
        "wc": "robust/raven.wc2014.1",
        "mix": "robust/stargazer.calib360.0",
        "calib360": "robust/stargazer.calib360.0"        
    },
    "chu": {
        "wc": "",
        "mix": "",
        "calib360": ""        
    },
    "our": {
        "wc": "fieldreg/eureka.wc2014.0",
        "mix": "fieldreg/stargazer.calib360.2",
        "calib360": "fieldreg/stargazer.calib360.2"        
    },    
}



class DataModule(InferDataModule):
    def __init__(self, dataset):
        self.dataset = dataset
                
    def get_inference_dataset(self) -> Dataset:
        return self.dataset


def create_datamodule(adapter_class, input_folder) -> InferDataModule:    
    dataset = adapter_class(
        dataset=ImageFolderDataset(
            folder=input_folder,
            search_pattern="*.png"
        ),
        noise_rotate=0,
        noise_translate=0,
        random_flip=False
    )        
    return DataModule(dataset=dataset)



def create_infermodule(method, scenario):
    adapter_class = RobustDatasetAdapter
    infermodule = None
            
    if (method == "gt"):
        infermodule = LabelInferModule()
    elif (method == "nie"):
        infermodule = RobustInferModule(
            BASE_PATH, EXPERIMENTS[method][scenario],
            make_images=True
        )
    elif (method == "chu"):
        infermodule = KpsfrInferModule(BASE_PATH, EXPERIMENTS[method][scenario])
        adapter_class = KpsfrDatasetAdapter
    elif (method == "theiner"):
        infermodule = TvCalibInferModule(
            segmentation_checkpoint=Path(
                ".scratch/models/segment_localization/train_59.pt"
            ),
            image_shape=(720, 1280),
            optim_steps=2000,
            make_images=True
        )
    elif (method == "our"):
        infermodule = FieldRegInferModule(
            BASE_PATH, EXPERIMENTS[method][scenario],
            make_images=True
        )
        adapter_class = partial(
            FieldRegDatasetAdapter,
            playfield_shape=(14,7),
            keep_original=True
        )

    return infermodule, adapter_class

def write_image(out_folder, idx, image):
    filepath = out_folder / f"{idx:05d}.png"        
    filepath.parent.mkdir(parents=True, exist_ok=True)        
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath.as_posix(), image)


def main(args):
    
    # Initialize inference
    infermodule, adapter_class = create_infermodule(args.method, args.scenario)
    datamodule = create_datamodule(adapter_class, args.input_dir)
    
    # Setup    
    infermodule.setup(datamodule)
    print(f"Infering Method @ Scenario: {args.method}, {args.scenario}")

    out_folder = Path(args.output_dir)

    # Go go!
    predictions_name = f"{args.method}-{args.scenario}.csv"
    with tqdm(
        datamodule.get_inference_dataset(),
        ascii=True,
        ncols=80
        ) as progress:        
        for idx, x in enumerate(progress):     
            p = infermodule.predict(x)
            write_image(out_folder, (idx+1), p["image_720p"])
            



if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--input_dir", "-i", default=".scratch/video/images", type=Path, help="Input image folder")
    p.add_argument("--output_dir", "-o", default=".scratch/video/output", type=Path, help="Output image folder")
    
    p.add_argument("--scenario", "-s", choices=["wc", "mix", "calib360"], default="wc", help="Scenario")
    p.add_argument("--method", "-m", choices=["gt", "nie", "chu", "theiner", "our"], default="our", help="Method to infer")

    main(p.parse_args())    