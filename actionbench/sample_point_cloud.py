# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def sample_point_cloud(
    point_cloud: torch.Tensor,
    n_pts: int,
    seed: int = 44,
) -> torch.Tensor:
    """
    Subsample points from a temporal point cloud sequence.

    Uses a single random permutation applied consistently across all timesteps
    to maintain point correspondence.

    Args:
        point_cloud: (T, N, C) point clouds over T timesteps.
        n_pts: Number of points to sample.
        seed: Random seed for reproducibility.

    Returns:
        (T, n_pts, C) subsampled point clouds.
    """
    n_pts_src = point_cloud.shape[1]
    if n_pts_src <= n_pts:
        return point_cloud

    rng = np.random.RandomState(seed=seed)
    indices = torch.from_numpy(rng.permutation(n_pts_src)[:n_pts]).long()
    return point_cloud[:, indices]
