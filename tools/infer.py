import argparse
from functools import partial
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from tqdm.auto import tqdm
from pathlib import Path
import torch

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


BASE_PATH = Path(".scratch/experiments")

EXPERIMENTS = {
    "nie": {
        "wc": "robust/raven.wc2014.1",
        "mix": "robust/stargazer.calib360.0",
        "calib360": "robust/stargazer.calib360.0"        
    },
    "chu": {
        "wc": "kpsfr/stargazer.wc2014.10",
        "mix": "kpsfr/yamato.calib360.0",
        "calib360": "kpsfr/yamato.calib360.0"        
    },
    "our": {
        "wc": "fieldreg/eureka.wc2014.0",
        "mix": "fieldreg/stargazer.calib360.2",
        "calib360": "fieldreg/stargazer.calib360.2"        
    },    
}


class LimitLengthDataset(torch.utils.data.Dataset):
    def __init__(self, ds, limit):
        self.ds = ds
        self.count = min(len(ds), limit)
            
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        return self.ds[idx]


class DataModule(InferDataModule):
    def __init__(self, dataset):
        self.dataset = dataset
                
    def get_inference_dataset(self) -> Dataset:
        return self.dataset




def create_datamodule(method, scenario, adapter_class) -> InferDataModule:    
    dataset = None    
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
        
        if (method == "our"):
            shape = (360, 640)
        else:
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



def create_infermodule(method, scenario):
    adapter_class = RobustDatasetAdapter
    infermodule = None
            
    if (method == "gt"):
        infermodule = LabelInferModule()
    elif (method == "nie"):
        infermodule = RobustInferModule(BASE_PATH, EXPERIMENTS[method][scenario])
    elif (method == "chu"):
        infermodule = KpsfrInferModule(BASE_PATH, EXPERIMENTS[method][scenario])
        #infermodule.load_pretrain(
        #    #filepath=".scratch/models/kpsfr/kpsfr_finetuned.pth"
        #    filepath=".scratch/models/kpsfr/kpsfr.pth"
        #)
        adapter_class = partial(
            KpsfrDatasetAdapter,
            num_objects=91
        )
    elif (method == "theiner"):
        infermodule = TvCalibInferModule(
            segmentation_checkpoint=Path(
                ".scratch/models/segment_localization/train_59.pt"
            ),
            image_shape=(720, 1280),
            optim_steps=2000                  
        )
    elif (method == "our"):
        infermodule = FieldRegInferModule(BASE_PATH, EXPERIMENTS[method][scenario])
        adapter_class = partial(
            FieldRegDatasetAdapter,
            playfield_shape=(14,7)
        )

    return infermodule, adapter_class


def main(args):
    
    # Initialize inference
    infermodule, adapter_class = create_infermodule(args.method, args.scenario)
    datamodule = create_datamodule(args.method, args.scenario, adapter_class)
    
    # Setup    
    infermodule.setup(datamodule)
    print(f"Infering Method @ Scenario: {args.method}, {args.scenario}")

    # Go go!
    predictions_name = f"{args.method}-{args.scenario}.csv"
    sink = PredictionsSink(target_filepath=args.inference_dir / predictions_name)
    ds = datamodule.get_inference_dataset()
    with tqdm(
        ds,
        ascii=True,
        ncols=80
        ) as progress:        
        for idx, x in enumerate(progress):     
            sink.write(infermodule.predict(x))            
            #if (idx >= 5): break
            
    sink.flush()



if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--inference_dir", "-id", default=".scratch/inference", type=Path, help="Folder to store inference into")
    p.add_argument("--scenario", "-s", choices=["wc", "mix", "calib360"], default="wc", help="Scenario")
    p.add_argument("--method", "-m", choices=["gt", "nie", "chu", "theiner", "our"], default="chu", help="Method to infer")

    main(p.parse_args())    