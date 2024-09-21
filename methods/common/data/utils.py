import cv2
import numpy as np
import random
import torch



def yards(x):
    return x * 1.0936132983



def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray.copy())
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray



def gen_template_grid():
    # === set uniform grid ===
    # field_dim_x, field_dim_y = 105.000552, 68.003928 # in meter
    field_dim_x, field_dim_y = 114.83, 74.37  # in yard
    # field_dim_x, field_dim_y = 115, 74 # in yard
    nx, ny = (13, 7)
    x = np.linspace(0, field_dim_x, nx)
    y = np.linspace(0, field_dim_y, ny)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    uniform_grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
    uniform_grid = np.concatenate((uniform_grid, np.ones(
        (uniform_grid.shape[0], 1))), axis=1)  # top2bottom, left2right
    # TODO: class label in template, each keypoints is (x, y, c), c is label that starts from 1
    for idx, pts in enumerate(uniform_grid):
        pts[2] = idx + 1  # keypoints label
    return uniform_grid


def put_lrflip_augmentation(frame, unigrid):

    frame_h, frame_w = frame.shape[0], frame.shape[1]
    flipped_img = np.fliplr(frame)

    # TODO: grid flipping and re-assign pixels class label, 1-91
    for ind, pts in enumerate(unigrid):
        pts[0] = frame_w - pts[0]
        col = (pts[2] - 1) // 7  # get each column of uniform grid
        pts[2] = pts[2] - (col - 6) * 2 * 7  # keypoints label
        
    return flipped_img, unigrid




def make_grid(images, nrow=4):
    num_images = len(images)
    ih, iw = images[0].shape[0], images[0].shape[1]
    rows = min(num_images, nrow)
    cols = (num_images + nrow-1) // nrow
    
    result = np.zeros(shape=(cols*ih, rows*iw, 3), dtype=np.uint8)
    for i in range(num_images):
        cell_x = i%nrow
        cell_y = i//nrow
        result[
            (cell_y+0)*ih:(cell_y+1)*ih,
            (cell_x+0)*iw:(cell_x+1)*iw
        ] = images[i]
        
    return result
