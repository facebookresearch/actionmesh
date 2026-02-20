# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    rotation_6d_to_matrix,
    transform3d,
)
from torch import nn
from torch.optim import Adam


def canonical_rotation_matrices() -> torch.Tensor:
    """
    Generate 24 rotation matrices for canonical orientations.

    Returns:
        (24, 3, 3) rotation matrices covering all axis-aligned orientations.
    """
    deg_to_rad = torch.pi / 180
    azim = (
        torch.tensor(
            [0] * 4 + [90] * 4 + [180] * 4 + [270] * 4 + [0] * 4 + [90] * 4,
            dtype=torch.float32,
        )
        * deg_to_rad
    )
    elev = (
        torch.tensor(
            [0] * 16 + [90] * 2 + [-90] * 2 + [90] * 2 + [-90] * 2,
            dtype=torch.float32,
        )
        * deg_to_rad
    )
    roll = (
        torch.tensor(
            [0, 90, 180, 270] * 4 + [0, 90] * 4,
            dtype=torch.float32,
        )
        * deg_to_rad
    )
    return euler_angles_to_matrix(
        torch.stack((azim, elev, roll), dim=-1), convention="XYZ"
    )


@torch.enable_grad()
def gradient_icp(
    pc_pred: torch.Tensor,
    pc_gt: torch.Tensor,
    lr: float = 0.01,
    n_iter: int = 200,
) -> transform3d.Transform3d:
    """
    Find the rigid + scale transformation from pc_pred to pc_gt.

    Uses gradient-based ICP with 24 canonical rotation initializations to
    avoid local minima. Optimizes rotation (6D representation), translation,
    and anisotropic scale.

    Args:
        pc_pred: (P, 3) predicted point cloud.
        pc_gt: (P, 3) ground truth point cloud.
        lr: Learning rate for Adam optimizer.
        n_iter: Number of optimization iterations.

    Returns:
        Transform3d that aligns pc_pred to pc_gt.
    """
    device = pc_pred.device

    R_init = canonical_rotation_matrices().to(device)
    n_rots = len(R_init)
    pc_pred = pc_pred[None].expand(n_rots, -1, -1)
    pc_gt = pc_gt[None].expand(n_rots, -1, -1)

    T = nn.Parameter(torch.zeros(n_rots, 3, device=device))
    r6d_init = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
    R_6d = nn.Parameter(r6d_init.repeat(n_rots, 1))
    s = nn.Parameter(torch.ones(n_rots, 3, device=device))

    opt = Adam(params=[T, R_6d, s], lr=lr)
    best_loss = float("inf")
    best_params = None

    for _ in range(n_iter):
        opt.zero_grad()
        R = R_init @ rotation_6d_to_matrix(R_6d)
        pc_transformed = s[:, None] * pc_pred @ R + T[:, None]
        loss = chamfer_distance(pc_transformed, pc_gt, batch_reduction=None)[0]
        loss.mean().backward()
        opt.step()

        min_loss, idx = loss.detach().min(0)
        if min_loss.item() < best_loss:
            best_loss = min_loss.item()
            best_params = (
                R[idx : idx + 1].detach().clone(),
                T[idx : idx + 1].detach().clone(),
                s[idx : idx + 1].detach().clone(),
            )

    Rf, Tf, sf = best_params
    return transform3d.Scale(sf).compose(
        transform3d.Rotate(Rf),
        transform3d.Translate(Tf),
    )
