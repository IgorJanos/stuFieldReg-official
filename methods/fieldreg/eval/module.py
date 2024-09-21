
import logging
import multiprocessing as mp
import numpy as np

from typing import Any
from tqdm.auto import tqdm

from methods.fieldreg.eval.base import EvalModule, EvalDataModule
from methods.fieldreg.eval.camera import Camera
from methods.fieldreg.eval.iou import iou_part, calc_iou_part
from methods.fieldreg.eval.utils import (
    batch_iterate, get_polylines, 
    get_polylines_homo, get_accuracy_at_threshold
)
from methods.common.data.utils import yards


def convert_homo(homo):
    if (homo is None): 
        return None
    
    # Homography is in evil yards, and relative to the playfield corner
    offset = np.array([
        [ 1.0, 0.0, -105/2.0 ],
        [ 0.0, 1.0, -68.0/2.0 ],
        [ 0, 0, 1]
    ])
    scale = np.array([
        [ 1.0/yards(1), 0, 0 ],
        [ 0, 1.0/yards(1), 0 ],
        [ 0, 0, 1]
    ])
    
    try:
        # homo robi pixle na yardy od rohu
        homo = offset @ scale @ homo
        homo = np.linalg.inv(homo)
    except:
        homo = None
    return homo


def process_act(batch_item):           
    # Unpack data
    gt_item, pred_item = batch_item        

    image_name = gt_item["item"]
    gt_homo = convert_homo(gt_item["homography"])
    pred_homo = convert_homo(pred_item["homography"])
           
    # Successful prediction ?
    if (pred_homo is None):
        result = {
            "name": image_name,
            "success": float(0.0),
            "ac@5": float(0.0),
            "ac@10": float(0.0),
            "ac@20": float(0.0),
        }
        return result

                
    IH, IW = 720, 1280
    
    # Compute Accuracy @ Threshold
    lines_gt = get_polylines_homo(gt_homo, (IH,IW))
    lines_p = get_polylines_homo(pred_homo, (IH,IW))

    ac_5, _ = get_accuracy_at_threshold(lines_gt, lines_p, threshold=5)
    ac_10, _ = get_accuracy_at_threshold(lines_gt, lines_p, threshold=10)
    ac_20, _ = get_accuracy_at_threshold(lines_gt, lines_p, threshold=20)
        
    result = {
        "name": image_name,
        "success": float(1.0),
        "ac@5": float(ac_5),
        "ac@10": float(ac_10),
        "ac@20": float(ac_20),
    }
    return result


def process_iou(batch_item):           
    # Unpack data
    gt_item, pred_item = batch_item        
    
    image_name = gt_item["item"]
    gt_homo = gt_item["homography"]
    pred_homo = pred_item["homography"]
           
    # Successful prediction ?
    if (pred_homo is None):
        result = {
            "name": image_name,
            "success": float(0.0),
            "iou_part": float(0.0)
        }
        return result
                
    IH, IW = 720, 1280
    
    template = np.zeros(shape=(1050, 680), dtype=np.uint8)
    frame = np.ones(shape=(720,1280,3), dtype=np.uint8)
    
    # Compute IOU
    if (True):
        iou, _, _, _ = calc_iou_part(
            pred_h=pred_homo,
            gt_h=gt_homo,
            frame=frame,
            template=template,
            frame_w=frame.shape[1],
            frame_h=frame.shape[0]
        )
    else:
        iou, image_iou = iou_part(
            h_pred=gt_homo, #pred_homo,
            h_true=gt_homo,
            image_height=IH,
            image_width=IW
        )                           
    
    result = {
        "name": image_name,
        "success": float(1.0),
        "iou_part": float(iou * 100.0)
    }
    return result



class EvalModule_ACt(EvalModule):
    def __init__(self, sink, batch_size):
        # Number of worker threads
        self.batch_size = batch_size
        self.sink = sink
           
           
    def setup(
        self,
        data_gt: EvalDataModule,
        data_pred: EvalDataModule
    ):
        self.data_gt = data_gt
        self.data_pred = data_pred
        
        if (len(self.data_gt.get_eval_dataset()) != len(self.data_pred.get_eval_dataset())):
            raise Exception(f"Dataset size mismatch!")
        
    
    
    def evaluate(self, logger):
        
        data_pair = zip(
            self.data_gt.get_eval_dataset(),
            self.data_pred.get_eval_dataset() 
        )       
        pool = mp.Pool(processes=self.batch_size)        

        with tqdm(
            iterable=batch_iterate(data_pair, batch_size=self.batch_size),
            ascii=True,
            ncols=80
        ) as progress:
            for batch in progress:                
                # Evaluate IOU
                result = pool.map(process_act, batch)                
                #result = [ process_act(b) for b in batch ]
                for item in result:
                    self.sink.write(item)
        
        df = self.sink.flush()
        
        # Print out statistics
        completeness = df["success"].mean()
        logger.info(f"Completeness: {completeness * 100.0}%")
        
        df_success = df[ df["success"] == 1.0 ]
        ac_mean = df_success[["ac@5", "ac@10", "ac@20"]].mean()
        
        logger.info(f"  AC @ 5 mean      : {ac_mean['ac@5']}")
        logger.info(f"  AC @ 10 mean     : {ac_mean['ac@10']}")
        logger.info(f"  AC @ 20 mean     : {ac_mean['ac@20']}")
                
        
    

class EvalModule_IOU(EvalModule):
    def __init__(self, sink):
        # Number of worker threads
        self.batch_size = 8
        self.sink = sink
           
           
    def setup(
        self,
        data_gt: EvalDataModule,
        data_pred: EvalDataModule
    ):
        self.data_gt = data_gt
        self.data_pred = data_pred
        
        if (len(self.data_gt.get_eval_dataset()) != len(self.data_pred.get_eval_dataset())):
            raise Exception(f"Dataset size mismatch!")
        
    
    
    def evaluate(self, logger):
        
        data_pair = zip(
            self.data_gt.get_eval_dataset(),
            self.data_pred.get_eval_dataset() 
        )       
        pool = mp.Pool(processes=self.batch_size)        

        with tqdm(
            iterable=batch_iterate(data_pair, batch_size=self.batch_size),
            ascii=True,
            ncols=80
        ) as progress:
            for batch in progress:                
                # Evaluate IOU
                result = pool.map(process_iou, batch) 
                #result = [ process_iou(b) for b in batch ]               
                for item in result:
                    self.sink.write(item)
        
        dataframe = self.sink.flush()
        
        # Print out statistics
        completeness = dataframe["success"].mean()
        logger.info(f"Completeness: {completeness * 100.0}%")
        dataframe_success = dataframe[ dataframe["success"] == 1.0 ]
        
        iou_mean = dataframe_success["iou_part"].mean()
        iou_median = dataframe_success["iou_part"].median()
        logger.info(f"IoU_part mean: {iou_mean}%")
        logger.info(f"IoU_part median: {iou_median}%")
        
    
    
    