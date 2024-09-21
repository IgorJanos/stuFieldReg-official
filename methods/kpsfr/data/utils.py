import cv2
import numpy as np
import random



def gen_im_whole_grid(mode, frame, f_idx, gt_homo, template, noise_trans, noise_rotate, index, vid_name=None):
    # === Warping image and grid for multi-frame method or cooredinate regression ===
    frame_w, frame_h = frame.shape[1], frame.shape[0]

    unigrid_copy = template.copy()  # (91, 3)
    unigrid_copy[:, 2] = 1

    gt_warp_grid = unigrid_copy @ np.linalg.inv(gt_homo.T)
    gt_warp_grid /= gt_warp_grid[:, 2, np.newaxis]

    # assign pixels class label, 1-91
    for idx, pts in enumerate(gt_warp_grid):
        pts[2] = idx + 1  # keypoints label

    # TODO: apply random small noise to the gt homography and the image is warp accordingly
    if mode == 'train' and (f_idx == 1 or f_idx == 2):  # hard level
        
        warp_image = None
        warp_grid = None
        
        # only store those points in image view
        l1, l2, label = [], [], []
        for pts, t_pts, sub_pts in zip(gt_warp_grid, unigrid_copy, template):
            if 0 <= pts[0] < frame_w and 0 <= pts[1] < frame_h:
                l1.append(pts)
                l2.append(t_pts)
                label.append(sub_pts[2])  # has labels
            else:
                l1.append([float('nan'), float('nan'), -1.])
                l2.append([float('nan'), float('nan'), 1.])
                label.append(-1.)

        src_grid = np.array(l1)
        tmp_grid = np.array(l2)
        class_labels = np.array(label)

        # TODO: do homography augmentation, around center??????
        center_x, center_y = frame_w / 2, frame_h / 2
        noise_scale = random.uniform(0.8, 1.05)
        scaling_mat = np.eye(3).astype(np.float32)
        scaling_mat[0, 0] = noise_scale
        scaling_mat[1, 1] = noise_scale
        if random.random() < 0.5:
            if random.random() < 0.5:
                scaling_mat[0, 2] = frame_h // 10
                scaling_mat[1, 2] = frame_h // 10
            else:
                scaling_mat[0, 2] = frame_h // 6
                scaling_mat[1, 2] = frame_h // 6
        tx = random.uniform(-noise_trans, noise_trans)
        ty = random.uniform(-noise_trans, noise_trans)
        translate_mat = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]], dtype=np.float32)

        theta = random.uniform(-noise_rotate, noise_rotate)
        deflection = 1.0
        theta = theta * 2.0 * deflection * np.pi  # in radians
        c, s = np.cos(theta), np.sin(theta)
        rotate_mat = np.array([[c, -s, 0],
                               [s, c, 0],
                               [0, 0, 1]], dtype=np.float32)

        pert_homo = rotate_mat @ gt_homo @ scaling_mat @ translate_mat @ rotate_mat.T
        pert_homo /= pert_homo[2, 2]

        # shape is (?, 3)
        if (tmp_grid.shape[0] > 0):
            pert_src_grid = tmp_grid @ np.linalg.inv(pert_homo.T)
            pert_src_grid /= pert_src_grid[:, 2, np.newaxis]

            for pts, cls in zip(pert_src_grid, class_labels):
                pts[2] = cls  # assign keypoints label

            src_list, dst_list = [], []
            for _src, _dst in zip(src_grid, pert_src_grid):
                if np.isnan(_dst).any():
                    continue
                # warp points maybe out of image resolution after perturbation
                if 0 <= _dst[0] < frame_w and 0 <= _dst[1] < frame_h:
                    src_list.append(_src)
                    dst_list.append(_dst)
                else:
                    _dst[0] = float('nan')
                    _dst[1] = float('nan')
                    _dst[2] = -1.
            src_pts = np.array(src_list)
            dst_pts = np.array(dst_list)
            if src_pts.shape[0] >= 4 and dst_pts.shape[0] >= 4:
                new_homo_mat, mask = cv2.findHomography(
                    src_pts[:, :2].reshape(-1, 1, 2), dst_pts[:, :2].reshape(-1, 1, 2), cv2.RANSAC, 5)
                if new_homo_mat is not None:
                    warp_image = cv2.warpPerspective(
                        frame, new_homo_mat, (frame_w, frame_h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
                    warp_grid = pert_src_grid.copy()
                    homo_mat = pert_homo
                else:
                    warp_image = None
                    warp_grid = None
            else:
                warp_image = None
                warp_grid = None
    else:
        warp_image = frame.copy()
        grid_list = []
        for ind, pts in enumerate(gt_warp_grid):
            if 0 <= pts[0] < frame_w and 0 <= pts[1] < frame_h:
                grid_list.append(pts)
            else:
                grid_list.append([float('nan'), float('nan'), -1.])
        warp_grid = np.array(grid_list)
        homo_mat = gt_homo

    if warp_image is None and warp_grid is None:
        warp_image = frame.copy()
        grid_list = []
        for ind, pts in enumerate(gt_warp_grid):
            if 0 <= pts[0] < frame_w and 0 <= pts[1] < frame_h:
                grid_list.append(pts)
            else:
                grid_list.append([float('nan'), float('nan'), -1.])
        warp_grid = np.array(grid_list)
        homo_mat = gt_homo

    return warp_image, warp_grid, homo_mat


