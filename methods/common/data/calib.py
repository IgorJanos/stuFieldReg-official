
import cv2
import numpy as np
import torch
import methods.common.data.utils as utils



class Calibrator:
    def __init__(
        self,
        num_classes: int=92,
        nms_threshold: float=0.995
    ):
        self.template_grid = utils.gen_template_grid()
        self.num_classes = num_classes
        self.nms_threshold = nms_threshold


    def find_homography(
        self, 
        heatmap_logits: torch.Tensor
    ):
        """ Extract keypoints from heatmap, and find homography matrix 
        
            heatmap_logits: torch.Tensor for individual frame (not a mini-batch!)
        """
        
        pred_rgb, pred_keypoints, scores_heatmap = self.decode_heatmap(heatmap_logits)
        homography = None
        
        # We need at least 4 point correspondences
        if (pred_rgb.shape[0] >= 4):
            src_pts, dst_pts = self.get_class_mapping(pred_rgb)
            
            # Find homography from point correspondences
            homography, _ = cv2.findHomography(
                src_pts.reshape(-1, 1, 2), 
                dst_pts.reshape(-1, 1, 2), 
                cv2.RANSAC, 
                10
            )
            
        return homography, pred_keypoints, scores_heatmap


    def decode_heatmap(self, heatmap_logits: torch.Tensor):
        """ Decode heatmap info keypoint set using non-maximum suppression        
            heatmap_logits: torc.Tensor with shape <NUM_CLASSES; H; W>
        """
        
        pred_heatmap = torch.softmax(heatmap_logits, dim=0)
        arg = torch.argmax(pred_heatmap, dim=0).detach().cpu().numpy()
        scores, pred_heatmap = torch.max(pred_heatmap, dim=0)
                
        # Convert to Numpy & get keypoints locations
        scores = scores.detach().cpu().numpy()
        pred_heatmap = pred_heatmap.detach().cpu().numpy()
        pred_class_dict = self.get_class_dict(scores, pred_heatmap)
        
        # Colorize
        num_classes = heatmap_logits.shape[0]
        np_scores = np.clip(arg * 255.0 / num_classes, 0, 255).astype(np.uint8)
        scores_heatmap = cv2.applyColorMap(np_scores, cv2.COLORMAP_HOT)
        scores_heatmap = cv2.cvtColor(scores_heatmap, cv2.COLOR_BGR2RGB)
        
        # Produce image with keypoints
        pred_keypoints = np.zeros_like(pred_heatmap, dtype=np.uint8)
        pred_rgb = []
        for _, (pk, pv) in enumerate(pred_class_dict.items()):
            if (pv):
                pred_keypoints[pv[1][0], pv[1][1]] = pk     # (H,W)
                # camera view point sets (x, y, label) in rgb domain not heatmap domain
                pred_rgb.append([pv[1][1] * 4, pv[1][0] * 4, pk])
        pred_rgb = np.asarray(pred_rgb, dtype=np.float32)  # (?, 3)
        
        # Return list of point locations, and image of keypoints
        return pred_rgb, pred_keypoints, scores_heatmap



    def get_class_mapping(self, rgb):
        src_pts = rgb.copy()
        cls_map_pts = []

        for ind, elem in enumerate(src_pts):
            coords = np.where(elem[2] == self.template_grid[:, 2])[0]  # find correspondence
            cls_map_pts.append(self.template_grid[coords[0]])
        dst_pts = np.array(cls_map_pts, dtype=np.float32)

        return src_pts[:, :2], dst_pts[:, :2]


    def get_class_dict(self, scores, pred):
        # Decode                               
        pred_cls_dict = {k: [] for k in range(1, self.num_classes)}
        for cls in range(1, self.num_classes):
            pred_inds = (pred == cls)

            # implies the current class does not appear in this heatmaps
            if not np.any(pred_inds):
                continue

            values = scores[pred_inds]
            max_score = values.max()
            max_index = values.argmax()

            indices = np.where(pred_inds)
            coords = list(zip(indices[0], indices[1]))

            # the only keypoint with max confidence is greater than threshold or not
            if max_score >= self.nms_threshold:
                pred_cls_dict[cls].append(max_score)
                pred_cls_dict[cls].append(coords[max_index])

        return pred_cls_dict
