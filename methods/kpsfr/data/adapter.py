
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import skimage.segmentation as ss

from project.data.transforms import NumpyToTensor

from methods.common.data.utils import to_torch
import methods.common.data.augmentation as aug

import methods.kpsfr.data.utils as utils
import random



class KpsfrDatasetAdapter(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        num_objects: int=4,
        noise_translate: float=0.0,
        noise_rotate: float=0.0,
        random_flip: bool=False,
        max_count = -1
    ):
        self.dataset = dataset
        self.num_objects = num_objects
        
        # Decide if augmentation is to be used
        if (noise_translate == 0.0 and noise_rotate == 0.0):
            self.augment = aug.WarpAugmentation(
                warp_function=utils.gen_im_whole_grid,
                mode="test",
                noise_translate=0.0,
                noise_rotate=0.0
            )
        else:
            self.augment = aug.WarpAugmentation(
                warp_function=utils.gen_im_whole_grid,
                mode="train",
                noise_translate=noise_translate,
                noise_rotate=noise_rotate
            )

        # Flip augmentation
        self.flip = aug.LeftRightFlipAugmentation(enabled=random_flip)  
        self.image_transform = transforms.Compose([
            NumpyToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.count = len(self.dataset)
        if (max_count > 0):
            self.count = min(self.count, max_count)

        
        
    def __len__(self):
        return self.count
    
    
    def __getitem__(self, idx):
        # Read from inner dataset
        sample = self.dataset[idx]
        
        # Produce new data with 3 samples
        result = self._get_data(
            sample, 
            num_items=3
        )                
        return result
    
    
    def _get_data(self, sample, num_items):        
        
        list_images = []
        list_homo = []
        list_heatmaps = []
        list_flip = []
                
        for f in range(num_items):           
            # Augment & Flip
            image, grid, homography = self.augment(sample["image"], sample["homography"], f)
            image, grid, is_flip = self.flip(image, grid)            
            
            # Append new image, homo and heatmap
            list_images.append(self.image_transform(image.copy()))
            list_homo.append(homography)
            list_heatmaps.append(self._get_heatmap(image, grid))
            list_flip.append(is_flip)
            
        list_heatmaps = np.stack(list_heatmaps, axis=0)         
        
        list_lookup = self._get_lookup_list(list_heatmaps, count=self.num_objects)
        list_selector, list_lookup = self._get_selector_list(list_lookup, count=self.num_objects)
        
        list_lookup_all = self._get_lookup_list(list_heatmaps, count=91)
        _, list_lookup_all = self._get_selector_list(list_lookup_all, count=91)
        
        list_target_heatmaps = self._get_target_heatmap(list_heatmaps, list_lookup)
        list_class_gt = self._get_class_ground_truth(list_target_heatmaps)
        
        result = {
            "rgb": torch.stack(list_images, dim=0),
            "target_dilated_hm": list_target_heatmaps,
            "cls_gt": list_class_gt,
            "homography": np.stack(list_homo, axis=0),
            "units": sample["units"],
            "selector": list_selector,
            "lookup": list_lookup,
            "lookup_gt": list_lookup_all,
            "flip": list_flip
        }
        
        result["image"] = result["rgb"][1]
        return result
        
        
    def _get_heatmap(self, image, grid):
        # Smaller resolution
        height = image.shape[0] // 4
        width = image.shape[1] // 4
        num_pts = grid.shape[0]
                
        heatmap = np.zeros((num_pts, height, width), dtype=np.float32)
        dilated_heatmap = np.zeros_like(heatmap)
        
        for idx in range(num_pts):
            if (np.isnan(grid[idx, 0]) or np.isnan(grid[idx, 1])):
                pass
            else:
                px = np.rint(grid[idx, 0] / 4).astype(np.int32)
                py = np.rint(grid[idx, 1] / 4).astype(np.int32)
                cls = int(grid[idx, 2]) - 1            
                if (0 <= px < width) and (0 <= py < height):
                    heatmap[cls][py, px] = grid[idx, 2]
                    dilated_heatmap[cls] = ss.expand_labels(
                        heatmap[cls], 
                        distance=5
                    )
                
        return dilated_heatmap
    
    
    def _get_target_heatmap(self, list_heatmaps, list_lookup):
        T, _, H, W = list_heatmaps.shape
        target_heatmap = torch.zeros((self.num_objects, T, H, W))
        num_items = list_heatmaps.shape[0]
        
        list_heatmaps = to_torch(list_heatmaps)        
        for f in range(num_items):
            for idx, obj in enumerate(list_lookup[f]):
                if obj != -1:
                    target_hm = list_heatmaps[f, int(obj)-1].clone()  # H*W
                    target_hm[target_hm == obj] = 1
                    target_heatmap[idx, f] = target_hm            
        
        return target_heatmap
    
    
    def _get_lookup_list(self, list_heatmaps, count):
        
        list_lookup = []
        
        num_items = list_heatmaps.shape[0]
        for f in range(num_items):
            labels = np.unique(list_heatmaps[0])
            labels = labels[labels != 0]    # remove background class
            lookup = np.ones(count, dtype=np.float32) * -1
            
            if len(labels) < 4:
                for idx, obj in enumerate(labels):
                    lookup[idx] = obj
            else:
                if (count == 91):
                    class_labels = np.ones(91, dtype=np.float32) * -1
                    for obj in labels:
                        class_labels[int(obj)-1]=obj
                    
                    sfp_interval = np.ones_like(class_labels)* -1
                    cls_id = np.unique(class_labels)
                    cls_id = cls_id[cls_id != -1]
                    cls_list = np.arange(cls_id.min(), cls_id.max()+1)
                    if (cls_list.min() > 10):
                        min_cls = cls_list.min()
                        l1 = np.arange(min_cls-10, min_cls)
                        cls_list = np.concatenate([l1,cls_list], axis=0)
                    if (cls_list.max() < 81):
                        max_cls = cls_list.max() + 1
                        l2 = np.arange(max_cls,max_cls+10)
                        cls_list = np.concatenate([cls_list,l2], axis=0)
                    for obj in cls_list:
                        lookup[int(obj)-1] = obj                
                    
                else:
                    for idx in range(count):
                        if len(labels) > 0:
                            target_object = random.choice(labels)
                            labels = labels[labels != target_object]
                            lookup[idx] = target_object

            list_lookup.append(lookup)
        
        return np.stack(list_lookup, axis=0)    # <T; CK:4>


    def _get_selector_list(self, list_lookup, count):
        
        if (list_lookup.shape[1] == 91):
            list_lookup = to_torch(list_lookup)
            selector_list = torch.ones_like(list_lookup)
            selector_list[list_lookup == -1] = 0
            return selector_list, list_lookup
        
        # Label reorder
        new_lookup_list = torch.ones((3, count)) * -1
        new_selector_list = torch.ones_like(new_lookup_list)

        inter01 = np.intersect1d(list_lookup[0], list_lookup[1])
        non_inter01 = np.setdiff1d(list_lookup[0], list_lookup[1])
        non_inter10 = np.setdiff1d(list_lookup[1], list_lookup[0])
        new0 = np.concatenate((inter01, non_inter01), axis=0)
        new1 = np.concatenate((inter01, non_inter10), axis=0)
        inter12, inter1_ind, _ = np.intersect1d(new1, list_lookup[2], return_indices=True)
        non_inter21 = np.setdiff1d(list_lookup[2], new1)
        
        if (new_lookup_list.shape[1] == new0.shape[0]):
            new_lookup_list[0, :] = to_torch(new0)
            
        if (new_lookup_list.shape[1] == new1.shape[0]):
            new_lookup_list[1, :] = to_torch(new1)
        
        if (inter12.shape[0] > 0):    
            new_lookup_list[2, inter1_ind] = to_torch(inter12)
                
        remain_ind = torch.where(new_lookup_list[2] == -1)[0]
        
        if (non_inter21.shape[0] > 0):
            new_lookup_list[2, remain_ind] = to_torch(non_inter21)
            
        new_selector_list[new_lookup_list == -1] = 0
        
        return new_selector_list, new_lookup_list
    
    
    def _get_class_ground_truth(self, list_target_heatmaps):
        T, C, H, W = list_target_heatmaps.shape
        cls_gt = torch.zeros((C,H,W))
        for f in range(C):
            for idx in range(self.num_objects):
                cls_gt[f][list_target_heatmaps[idx,f] == 1] = idx + 1
                
        return cls_gt
        