import argparse
import logging
import multiprocessing as mp

from tqdm.auto import tqdm
from pathlib import Path

from methods.fieldreg.eval.datamodule import EvalDataModule_Calib360
from methods.fieldreg.eval.module import EvalModule_IOU, EvalModule_ACt
from methods.fieldreg.eval.sink import EvaluationSink_IOU, EvaluationSink_ACt

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)





def get_predictions_filename(dataset, method):
    return f"pred-{dataset}-{method}.csv"



def main(args):
    
    # Split inference file into method and scenario
    method, scenario = args.inference_file.stem.split("-")
    
    # Setup reports
    args.reports_dir.mkdir(parents=True, exist_ok=True)   
    reports_fn = f"{method}-{scenario}-{args.metric}.txt"
    
    # Initialize logger
    fh = logging.FileHandler((args.reports_dir / reports_fn).as_posix())
    fh.setLevel(logging.DEBUG) 
    logger.addHandler(fh)    
    logger.info("------ Starting evaluation:")    
    logger.info(f"  Running evaluation - {method} - {scenario} - {args.metric}")
    
    # Get our filenames
    gt_filename = args.inference_file.parent / f"gt-{scenario}.csv"
    # Setup data modules
    data_gt = EvalDataModule_Calib360(gt_filename)
    data_pred = EvalDataModule_Calib360(args.inference_file)
    
    # Metric
    if (args.metric == "iou"):
        reports_csv_fn = f"{method}-{scenario}-{args.metric}.csv"
        evalmodule = EvalModule_IOU(
            sink=EvaluationSink_IOU(
                target_filepath=args.reports_dir / reports_csv_fn
            )
        )
    elif (args.metric == "act"):
        reports_csv_fn = f"{method}-{scenario}-{args.metric}.csv"
        evalmodule = EvalModule_ACt(
            sink=EvaluationSink_ACt(
                target_filepath=args.reports_dir / reports_csv_fn
            ),
            batch_size=mp.cpu_count()
        )
    else:
        raise Exception(f"Invalid metric: {args.metric}")
    
    # Go go !
    evalmodule.setup(data_gt, data_pred)
    evalmodule.evaluate(logger)






if __name__ == "__main__":
    p = argparse.ArgumentParser()    
    
    p.add_argument(
        "--inference_file", "-i", 
        default=".scratch/inference/theiner-wc.csv",
        type=Path, 
        help="Input inference CSV file"
    )
    p.add_argument("--reports_dir", "-rd", default=".scratch/reports", type=Path, help="Folder to store reports into")
    p.add_argument("--metric", "-mt", choices=["iou", "act"], default="act", help="Metric to evaluate")
    
    main(p.parse_args())