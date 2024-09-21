import cv2
import numpy as np
import random


def resize_homo(image, homo, target_shape=(720, 1280), interpolation=cv2.INTER_LINEAR):
    H,W = image.shape[0], image.shape[1]
    TH,TW = target_shape
    
    if ((TH==H) and (TW==W)):
        return image, homo
    
    out_image = cv2.resize(image, (TW,TH), interpolation=interpolation)        
    scale = np.array([
        [ W/TW, 0, 0 ],
        [ 0, H/TH, 0 ],
        [ 0, 0, 1]
    ])
    
    out_homo = homo
    if (homo is not None):
        out_homo = homo @ scale
    return out_image, out_homo


