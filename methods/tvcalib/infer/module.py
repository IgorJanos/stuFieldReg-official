import torch
import torch.nn as nn
from functools import partial

from pathlib import Path
from typing import Any, Dict, Tuple
from methods.common.infer.base import *
from methods.common.infer.base import InferDataModule
import project as p

import torchvision.transforms as T
from methods.tvcalib.sn_segmentation.src.custom_extremities import (
    generate_class_synthesis, get_line_extremities
)
from methods.tvcalib.models.segmentation import InferenceSegmentationModel
from methods.tvcalib.data.dataset import InferenceDatasetCalibration
from methods.tvcalib.data.utils import custom_list_collate
from methods.tvcalib.cam_modules import CameraParameterWLensDistDictZScore, SNProjectiveCamera
from methods.tvcalib.utils.linalg import distance_line_pointcloud_3d, distance_point_pointcloud
from methods.tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from methods.tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from methods.tvcalib.utils.io import detach_dict, tensor2list
from methods.common.data.utils import yards

from kornia.geometry.conversions import convert_points_to_homogeneous
from tqdm.auto import tqdm

from methods.robust.loggers.preview import RobustPreviewLogger



class TvCalibInferModule(InferModule):
    def __init__(
        self,
        segmentation_checkpoint: Path,
        image_shape=(720,1280),
        optim_steps=2000,
        lens_dist: bool=False,
        playfield_size=(105, 68),
        make_images: bool=False

    ):
        self.image_shape = image_shape
        self.device = torch.device("cuda")
        self.make_images = make_images
        
        # We use the logger to draw visualizations
        self.previewer = RobustPreviewLogger(
            None, num_images=1
        )
        
        self.fn_generate_class_synthesis = partial(
            generate_class_synthesis, 
            radius=4
        )
        self.fn_get_line_extremities = partial(
            get_line_extremities, 
            maxdist=30, 
            width=455, 
            height=256, 
            num_points_lines=4, 
            num_points_circles=8
        )
        
        # Segmentation model
        self.model_seg = InferenceSegmentationModel(
            segmentation_checkpoint,
            self.device
        )

        self.object3d = SoccerPitchLineCircleSegments(
            device=self.device, 
            base_field=SoccerPitchSNCircleCentralSplit()
        )
        self.object3dcpu = SoccerPitchLineCircleSegments(
            device="cpu", 
            base_field=SoccerPitchSNCircleCentralSplit()
        )

        # Calibration module 
        batch_size_calib = 1       
        self.model_calib = TVCalibModule(
            self.object3d,
            get_cam_distr(1.96, batch_size_calib, 1),
            get_dist_distr(batch_size_calib, 1) if lens_dist else None,
            (image_shape[0], image_shape[1]),
            optim_steps,
            self.device,
            log_per_step=False,
            tqdm_kwqargs=None,
        )
        self.resize = T.Compose([
            T.Resize(size=(256,455))
        ])
        self.offset = np.array([
            [1, 0, playfield_size[0]/2.0 ],
            [0, 1, playfield_size[1]/2.0 ],
            [0, 0, 1]
        ])

    
    
    def setup(self, datamodule: InferDataModule):
        pass
    
    
    def predict(self, x: Any) -> Dict:
        
        """
            1. Run segmentation & Pick keypoints
            2. Calibrate based on selected points
        """
        
        # Segment
        image = x["image"]        
        keypoints = self._segment(x["image"])
        
        # Calibrate
        homo = self._calibrate(keypoints)
        
        # Rescale to 720p
        image_720p = self.previewer.to_image(image.clone().detach().cpu())
        
         # Draw predicted playing field
        if (homo is not None):            
            # to yards
            to_yards = np.array([
                [ yards(1.0), 0, 0 ],
                [ 0, yards(1.0), 0 ],
                [ 0, 0, 1]
            ])
            #homo = to_yards @ homo
                        
            try:
                inv_homo = np.linalg.inv(homo) @ self.previewer.scale
                image_720p = self.previewer.draw_playfield(
                    image_720p, 
                    self.previewer.image_playfield, 
                    inv_homo,
                    color=(255,0,0), alpha=1.0,
                    flip=False
                )
            except:
                # Homography might
                pass
        
        result = {
            "homography": homo
        }
        
        if (self.make_images):
            result["image_720p"] = image_720p
        
        return result
    
    
    def _segment(self, image):
        
        # Image -> <1;3;256;455>
        image = self.resize(image)                
        with torch.no_grad():
            sem_lines = self.model_seg.inference(
                image.unsqueeze(0).to(self.device)
            )
        # <B;256;455>
        sem_lines = sem_lines.detach().cpu().numpy().astype(np.uint8)
            
        # Point selection
        skeletons_batch = self.fn_generate_class_synthesis(sem_lines[0])
        keypoints_raw_batch = self.fn_get_line_extremities(skeletons_batch)
        
        # Return the keypoints
        return keypoints_raw_batch
    
    
    def _calibrate(self, keypoints):
        
        # Just wrap around the keypoints
        ds = InferenceDatasetCalibration(
            [keypoints],
            self.image_shape[1], self.image_shape[0],
            self.object3d
        )
        
        # Get the first item and optimize it
        _batch_size = 1
        x_dict = custom_list_collate([ds[0]])
        per_sample_loss, cam, _ = self.model_calib.self_optim_batch(x_dict)
        output_dict = tensor2list(
            detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss})
        )
        
        homo = output_dict["homography"][0]
        if (len(homo) > 0):
            homo = np.array(homo[0])

            to_yards = np.array([
                [ yards(1), 0, 0 ],
                [ 0, yards(1), 0 ],
                [ 0, 0, 1]
            ])

            # Shift the homography by half the playing field
            homo = to_yards @ self.offset @ homo
                        
        else:
            homo = None
        return homo
    
    
    
    
    
    
    
class TVCalibModule(torch.nn.Module):
    def __init__(
        self,
        model3d,
        cam_distr,
        dist_distr,
        image_dim: Tuple[int, int],
        optim_steps: int,
        device="cpu",
        tqdm_kwqargs=None,
        log_per_step=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_height, self.image_width = image_dim
        self.principal_point = (self.image_width / 2, self.image_height / 2)
        self.model3d = model3d
        self.cam_param_dict = CameraParameterWLensDistDictZScore(
            cam_distr, dist_distr, device=device
        )

        self.lens_distortion_active = False if dist_distr is None else True
        self.optim_steps = optim_steps
        self._device = device

        self.optim = torch.optim.AdamW(
            self.cam_param_dict.param_dict.parameters(), lr=0.1, weight_decay=0.01
        )
        self.Scheduler = partial(
            torch.optim.lr_scheduler.OneCycleLR,
            max_lr=0.05,
            total_steps=self.optim_steps,
            pct_start=0.5,
        )

        if self.lens_distortion_active:
            self.optim_lens_distortion = torch.optim.AdamW(
                self.cam_param_dict.param_dict_dist.parameters(), lr=1e-3, weight_decay=0.01
            )
            self.Scheduler_lens_distortion = partial(
                torch.optim.lr_scheduler.OneCycleLR,
                max_lr=1e-3,
                total_steps=self.optim_steps,
                pct_start=0.33,
                optimizer=self.optim_lens_distortion,
            )

        self.tqdm_kwqargs = tqdm_kwqargs
        if tqdm_kwqargs is None:
            self.tqdm_kwqargs = {}

        self.hparams = {"optim": str(self.optim), "scheduler": str(self.Scheduler)}
        self.log_per_step = log_per_step

    def forward(self, x):

        # individual camera parameters & distortion parameters
        phi_hat, psi_hat = self.cam_param_dict()

        cam = SNProjectiveCamera(
            phi_hat,
            psi_hat,
            self.principal_point,
            self.image_width,
            self.image_height,
            device=self._device,
            nan_check=False,
        )

        # (batch_size, num_views_per_cam, 3, num_segments, num_points)
        points_px_lines_true = x["lines__ndc_projected_selection_shuffled"].to(self._device)
        batch_size, T_l, _, S_l, N_l = points_px_lines_true.shape

        # project circle points
        points_px_circles_true = x["circles__ndc_projected_selection_shuffled"].to(self._device)
        _, T_c, _, S_c, N_c = points_px_circles_true.shape
        assert T_c == T_l

        ####################  line-to-point distance at pixel space ####################
        # start and end point (in world coordinates) for each line segment
        points3d_lines_keypoints = self.model3d.line_segments  # (3, S_l, 2) to (S_l * 2, 3)
        points3d_lines_keypoints = points3d_lines_keypoints.reshape(3, S_l * 2).transpose(0, 1)
        points_px_lines_keypoints = convert_points_to_homogeneous(
            cam.project_point2ndc(points3d_lines_keypoints, lens_distortion=False)
        )  # (batch_size, t_l, S_l*2, 3)

        if batch_size < cam.batch_dim:  # actual batch_size smaller than expected, i.e. last batch
            points_px_lines_keypoints = points_px_lines_keypoints[:batch_size]

        points_px_lines_keypoints = points_px_lines_keypoints.view(batch_size, T_l, S_l, 2, 3)

        lp1 = points_px_lines_keypoints[..., 0, :].unsqueeze(-2)  # -> (batch_size, T_l, 1, S_l, 3)
        lp2 = points_px_lines_keypoints[..., 1, :].unsqueeze(-2)  # -> (batch_size, T_l, 1, S_l, 3)
        # (batch_size, T, 3, S, N) -> (batch_size, T, 3, S*N) -> (batch_size, T, S*N, 3) -> (batch_size, T, S, N, 3)
        pc = (
            points_px_lines_true.view(batch_size, T_l, 3, S_l * N_l)
            .transpose(2, 3)
            .view(batch_size, T_l, S_l, N_l, 3)
        )

        if self.lens_distortion_active:
            # undistort given points
            pc = pc.view(batch_size, T_l, S_l * N_l, 3)
            pc = pc.detach().clone()
            pc[..., :2] = cam.undistort_points(
                pc[..., :2], cam.intrinsics_ndc, num_iters=1
            )  # num_iters=1 might be enough for a good approximation
            pc = pc.view(batch_size, T_l, S_l, N_l, 3)

        distances_px_lines_raw = distance_line_pointcloud_3d(
            e1=lp2 - lp1, r1=lp1, pc=pc, reduce=None
        )  # (batch_size, T_l, S_l, N_l)
        distances_px_lines_raw = distances_px_lines_raw.unsqueeze(-3)
        # (..., 1, S_l, N_l,), i.e. (batch_size, T, 1, S_l, N_l)
        ####################  circle-to-point distance at pixel space ####################

        # circle segments are approximated as point clouds of size N_c_star
        points3d_circles_pc = self.model3d.circle_segments
        _, S_c, N_c_star = points3d_circles_pc.shape
        points3d_circles_pc = points3d_circles_pc.reshape(3, S_c * N_c_star).transpose(0, 1)
        points_px_circles_pc = cam.project_point2ndc(points3d_circles_pc, lens_distortion=False)

        if batch_size < cam.batch_dim:  # actual batch_size smaller than expected, i.e. last batch
            points_px_circles_pc = points_px_circles_pc[:batch_size]

        if self.lens_distortion_active:
            # (batch_size, T_c, _, S_c, N_c)
            points_px_circles_true = points_px_circles_true.view(
                batch_size, T_c, 3, S_c * N_c
            ).transpose(2, 3)
            points_px_circles_true = points_px_circles_true.detach().clone()
            points_px_circles_true[..., :2] = cam.undistort_points(
                points_px_circles_true[..., :2], cam.intrinsics_ndc, num_iters=1
            )
            points_px_circles_true = points_px_circles_true.transpose(2, 3).view(
                batch_size, T_c, 3, S_c, N_c
            )

        distances_px_circles_raw = distance_point_pointcloud(
            points_px_circles_true, points_px_circles_pc.view(batch_size, T_c, S_c, N_c_star, 2)
        )

        distances_dict = {
            "loss_ndc_lines": distances_px_lines_raw,  # (batch_size, T_l, 1, S_l, N_l)
            "loss_ndc_circles": distances_px_circles_raw,  # (batch_size, T_c, 1, S_c, N_c)
        }
        return distances_dict, cam

    def self_optim_batch(self, x, *args, **kwargs):

        scheduler = self.Scheduler(self.optim)  # re-initialize lr scheduler for every batch
        if self.lens_distortion_active:
            scheduler_lens_distortion = self.Scheduler_lens_distortion()

        # TODO possibility to init from x; -> modify dataset that yields x
        self.cam_param_dict.initialize(None)
        self.optim.zero_grad()
        if self.lens_distortion_active:
            self.optim_lens_distortion.zero_grad()

        keypoint_masks = {
            "loss_ndc_lines": x["lines__is_keypoint_mask"].to(self._device),
            "loss_ndc_circles": x["circles__is_keypoint_mask"].to(self._device),
        }
        num_actual_points = {
            "loss_ndc_circles": keypoint_masks["loss_ndc_circles"].sum(dim=(-1, -2)),
            "loss_ndc_lines": keypoint_masks["loss_ndc_lines"].sum(dim=(-1, -2)),
        }
        # print({f"{k} {v}" for k, v in num_actual_points.items()})

        per_sample_loss = {}
        per_sample_loss["mask_lines"] = keypoint_masks["loss_ndc_lines"]
        per_sample_loss["mask_circles"] = keypoint_masks["loss_ndc_circles"]

        per_step_info = {"loss": [], "lr": []}
        # with torch.autograd.detect_anomaly():
        with tqdm(range(self.optim_steps), **self.tqdm_kwqargs) as pbar:
            for step in pbar:

                self.optim.zero_grad()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.zero_grad()

                # forward pass
                distances_dict, cam = self(x)

                # create mask for batch dimension indicating whether distances and loss are computed
                # based on per-sample distance

                # distance calculate with masked input and output
                losses = {}
                for key_dist, distances in distances_dict.items():
                    # for padded points set distance to 0.0
                    # then sum over dimensions (S, N) and divide by number of actual given points
                    distances[~keypoint_masks[key_dist]] = 0.0

                    # log per-point distance
                    per_sample_loss[f"{key_dist}_distances_raw"] = distances

                    # sum px distance over S and number of points, then normalize given the number of annotations
                    distances_reduced = distances.sum(dim=(-1, -2))  # (B, T, 1, S, M) -> (B, T, 1)
                    distances_reduced = distances_reduced / num_actual_points[key_dist]

                    # num_actual_points == 0 -> set loss for this segment to 0.0 to prevent division by zero
                    distances_reduced[num_actual_points[key_dist] == 0] = 0.0

                    distances_reduced = distances_reduced.squeeze(-1)  # (B, T, 1) -> (B, T,)
                    per_sample_loss[key_dist] = distances_reduced

                    loss = distances_reduced.mean(dim=-1)  # mean over T dimension: (B, T, )-> (B,)
                    # only relevant for autograd:
                    # sum over batch dimension
                    # --> different batch sizes do not change the per sample loss and its gradients
                    loss = loss.sum()

                    losses[key_dist] = loss

                # each segment and annotation contributes equal to the loss -> no need for weighting segment types
                loss_total_dist = losses["loss_ndc_lines"] + losses["loss_ndc_circles"]
                loss_total = loss_total_dist

                if self.log_per_step:
                    per_step_info["lr"].append(scheduler.get_last_lr())
                    per_step_info["loss"].append(distances_reduced)  # log per sample loss
                if step % 50 == 0:
                    pbar.set_postfix(
                        loss=f"{loss_total_dist.detach().cpu().tolist():.5f}",
                        loss_lines=f'{losses["loss_ndc_lines"].detach().cpu().tolist():.3f}',
                        loss_circles=f'{losses["loss_ndc_circles"].detach().cpu().tolist():.3f}',
                    )

                loss_total.backward()
                self.optim.step()
                scheduler.step()
                if self.lens_distortion_active:
                    self.optim_lens_distortion.step()
                    scheduler_lens_distortion.step()

        per_sample_loss["loss_ndc_total"] = torch.sum(
            torch.stack([per_sample_loss[key_dist] for key_dist in distances_dict.keys()], dim=0),
            dim=0,
        )

        if self.log_per_step:
            per_step_info["loss"] = torch.stack(
                per_step_info["loss"], dim=-1
            )  # (n_steps, batch_dim, temporal_dim)
            per_step_info["lr"] = torch.tensor(per_step_info["lr"])
        return per_sample_loss, cam, per_step_info
