# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import trimesh
from chamfer import compute_chamfer_score, compute_motion_chamfer_score
from icp import gradient_icp
from sample_mesh import sample_meshes
from sample_point_cloud import sample_point_cloud
from tqdm import tqdm


def _compute_per_frame_icp(
    gt_pc: torch.Tensor,
    pred_pc: torch.Tensor,
) -> list:
    """
    Compute per-frame ICP transforms between predicted and GT point clouds.

    Args:
        gt_pc: (T, N, 3) ground truth point clouds.
        pred_pc: (T, N, 3) predicted point clouds.

    Returns:
        List of T Transform3d objects aligning pred to gt per frame.
    """
    n_ts = gt_pc.shape[0]
    transforms = []
    for k in tqdm(range(n_ts), desc="ICP per-frame"):
        transforms.append(
            gradient_icp(
                pc_gt=gt_pc[k],
                pc_pred=pred_pc[k],
                lr=0.01,
                n_iter=200,
            )
        )
    return transforms


def _compute_unified_icp(
    gt_pc: torch.Tensor,
    pred_pc: torch.Tensor,
) -> object:
    """
    Compute a single ICP transform using only the first frame.

    Args:
        gt_pc: (T, N, 3) ground truth point clouds.
        pred_pc: (T, N, 3) predicted point clouds.

    Returns:
        Transform3d aligning pred to gt based on first frame only.
    """
    return gradient_icp(
        pc_gt=gt_pc[0],
        pc_pred=pred_pc[0],
        lr=0.01,
        n_iter=200,
    )


def compute_chamfer_3d_4d(
    gt_pc: torch.Tensor,
    pred_meshes: list[trimesh.Trimesh],
    device: str,
    is_4D: bool = False,
    n_pts_icp: int = 10_000,
    n_pts_chamfer: int = 100_000,
    seed: int = 44,
) -> tuple[float, float, float]:
    """
    Compute 3D and 4D Chamfer distance between meshes and GT points.

    Performs ICP alignment followed by Chamfer distance computation. Computes
    two variants:
        - cd_3d: Per-frame ICP alignment, then average Chamfer across frames.
        - cd_4d: Unified ICP from first frame, then average Chamfer.

    For 4D evaluation, also computes motion-aware Chamfer using synchronized
    sampling.

    Args:
        gt_pc: (T, N, 3) ground truth point clouds over T timesteps.
        pred_meshes: List of T predicted meshes.
        device: Device for computation ("cuda" or "cpu").
        is_4D: If True, compute motion Chamfer distance.
        n_pts_icp: Number of points for ICP alignment.
        n_pts_chamfer: Number of points for Chamfer computation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (cd_3d, cd_4d, cd_motion) where:
            - cd_3d: Chamfer with per-frame ICP alignment.
            - cd_4d: Chamfer with unified (first-frame) ICP alignment.
            - cd_motion: Motion Chamfer (0.0 if is_4D is False).
    """
    n_ts = len(pred_meshes)

    pred_pc = sample_meshes(
        pred_meshes, n_pts=n_pts_chamfer, synchronized=False, seed=seed
    )
    pred_pc_icp = sample_point_cloud(pred_pc, n_pts=n_pts_icp, seed=seed)
    gt_pc_icp = sample_point_cloud(gt_pc, n_pts=n_pts_icp, seed=seed)

    pred_pc = pred_pc.to(device)
    gt_pc = gt_pc.to(device)
    pred_pc_icp = pred_pc_icp.to(device)
    gt_pc_icp = gt_pc_icp.to(device)

    # Per-frame ICP alignment
    icp_list = _compute_per_frame_icp(gt_pc_icp, pred_pc_icp)
    icp_transforms_3D = icp_list[0].stack(*icp_list[1:])

    # Unified ICP alignment (first frame only)
    icp_transforms_u4D = _compute_unified_icp(gt_pc_icp, pred_pc_icp)

    # Transform predicted point clouds
    pred_pc_aligned_3D = icp_transforms_3D.transform_points(pred_pc)
    pred_pc_aligned_u4D = icp_transforms_u4D.transform_points(pred_pc)

    # Compute Chamfer distances
    cd_3d = np.mean(
        [
            compute_chamfer_score(gt=gt_pc[k].cpu(), pred=pred_pc_aligned_3D[k].cpu())
            for k in range(n_ts)
        ]
    )
    cd_4d = np.mean(
        [
            compute_chamfer_score(gt=gt_pc[k].cpu(), pred=pred_pc_aligned_u4D[k].cpu())
            for k in range(n_ts)
        ]
    )

    cd_motion = 0.0
    if is_4D:
        pred_pc_4D = sample_meshes(
            pred_meshes, n_pts=n_pts_chamfer, synchronized=True, seed=seed
        )
        pred_pc_4D = pred_pc_4D.to(device)
        pred_pc_aligned_4D = icp_transforms_u4D.transform_points(pred_pc_4D)

        cd_motion = compute_motion_chamfer_score(
            preds=pred_pc_aligned_4D.cpu(),
            gts=gt_pc.cpu(),
        )

    return cd_3d, cd_4d, cd_motion
