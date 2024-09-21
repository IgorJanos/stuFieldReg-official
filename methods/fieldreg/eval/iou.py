import kornia
import numpy as np
import torch
import scipy.ndimage
import cv2

def _warp_ones2_sn_template(H, image_height: int, image_width: int):
    # H that maps from image to template
    # translate center of the homography matrix to image origin (upper left)
    H = torch.from_numpy(H).float()
    H = H.unsqueeze(0).repeat(1, 1, 1)
    H = H / H[:, -1, -1]  # normalize homography

    T = torch.eye(3).unsqueeze(0).float()
    T[:, 0, -1] = 105 / 2
    T[:, 1, -1] = 68 / 2
    H = T @ H @ T
    warped_top = kornia.geometry.transform.homography_warp(
        torch.ones(size=(1, 1, image_height, image_width)),
        H,
        dsize=(68, 105),
        normalized_homography=False,
        normalized_coordinates=False,
    )
    warped_top = torch.from_numpy(scipy.ndimage.binary_fill_holes(warped_top))
    return warped_top.to(torch.uint8)



def iou_part(h_pred, h_true, image_height: int, image_width: int, eps=1e-6):


    output_mask = _warp_ones2_sn_template(h_pred, image_height, image_width) # (1, 1, 68, 105)
    target_mask = _warp_ones2_sn_template(h_true, image_height, image_width) # (1, 1, 68, 105)

    output_mask[output_mask > 0] = 1
    target_mask[target_mask > 0] = 1

    intersection_mask = output_mask * target_mask
    output = output_mask.sum(dim=[1, 2, 3])
    target = target_mask.sum(dim=[1, 2, 3])
    intersection = intersection_mask.sum(dim=[1, 2, 3])
    union = output + target - intersection
    iou = intersection / (union + eps)

    img_composite = torch.zeros((1, 3, 68, 105), dtype=torch.uint8)
    img_composite[:, 0] = target_mask[:, 0]
    img_composite[:, 2] = output_mask[:, 0]
    img_composite[(img_composite.sum(dim=1) == 2.0).unsqueeze(1).repeat(1, 3, 1, 1)] = 1.0
    img_composite = (img_composite * 255.0).to(torch.uint8)

    img_composite = img_composite[0].cpu().detach().numpy()
    img_composite = np.transpose(img_composite, (1,2,0))
    
    return iou, img_composite


def calc_iou_part(pred_h, gt_h, frame, template, frame_w=1280, frame_h=720, template_w=115, template_h=74):

    # TODO: calculate iou part
    # === render ===
    render_h, render_w = template.shape  # (1050, 680)
    dst = np.array(template)

    # Create three channels (680, 1050, 3)
    dst = np.stack((dst, ) * 3, axis=-1)

    scaling_mat = np.eye(3)
    scaling_mat[0, 0] = render_w / template_w
    scaling_mat[1, 1] = render_h / template_h

    frame = np.uint8(frame * 255)  # 0-1 map to 0-255
    gt_mask_render = cv2.warpPerspective(
        frame, scaling_mat @ gt_h, (render_w, render_h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    pred_mask_render = cv2.warpPerspective(
        frame, scaling_mat @ pred_h, (render_w, render_h), cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    # === blending ===
    dstf = dst.astype(float) / 255
    gt_mask_renderf = gt_mask_render.astype(float) / 255
    gt_resultf = cv2.addWeighted(dstf, 0.3, gt_mask_renderf, 0.7, 0.0)
    gt_result = np.uint8(gt_resultf * 255)
    pred_mask_renderf = pred_mask_render.astype(float) / 255
    pred_resultf = cv2.addWeighted(dstf, 0.3, pred_mask_renderf, 0.7, 0.0)
    pred_result = np.uint8(pred_resultf * 255)

    # field template binary mask
    field_mask = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255
    gt_mask = cv2.warpPerspective(field_mask, gt_h, (template_w, template_h),
                                  cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    pred_mask = cv2.warpPerspective(field_mask, pred_h, (template_w, template_h),
                                    cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
    gt_mask[gt_mask > 0] = 255
    pred_mask[pred_mask > 0] = 255

    intersection = ((gt_mask > 0) * (pred_mask > 0)).sum()
    union = (gt_mask > 0).sum() + (pred_mask > 0).sum() - intersection

    if union <= 0:
        print('part union', union)
        # iou = float('nan')
        iou = 0.
    else:
        iou = float(intersection) / float(union)

    # === blending ===
    gt_white_area = (gt_mask[:, :, 0] == 255) & (
        gt_mask[:, :, 1] == 255) & (gt_mask[:, :, 2] == 255)
    gt_fill = gt_mask.copy()
    gt_fill[gt_white_area, 0] = 255
    gt_fill[gt_white_area, 1] = 0
    gt_fill[gt_white_area, 2] = 0
    pred_white_area = (pred_mask[:, :, 0] == 255) & (
        pred_mask[:, :, 1] == 255) & (pred_mask[:, :, 2] == 255)
    pred_fill = pred_mask.copy()
    pred_fill[pred_white_area, 0] = 0
    pred_fill[pred_white_area, 1] = 255
    pred_fill[pred_white_area, 2] = 0
    gt_maskf = gt_fill.astype(float) / 255
    pred_maskf = pred_fill.astype(float) / 255
    fill_resultf = cv2.addWeighted(gt_maskf, 0.5,
                                   pred_maskf, 0.5, 0.0)
    fill_result = np.uint8(fill_resultf * 255)

    return iou, gt_result, pred_result, fill_result