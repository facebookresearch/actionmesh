# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from itertools import cycle
from typing import Optional

import numpy as np
import torch
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    look_at_rotation,
    PerspectiveCameras,
)

# Module-level constants for coordinate system conversions
_M_CAM_FLIP = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
_M_Y_UP_P3D_TO_BLENDER = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
_M_Y_UP_BLENDER_TO_P3D = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def _convert_extrinsics(
    R: np.ndarray,
    T: np.ndarray,
    M_cam: np.ndarray,
    M_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic extrinsic conversion between coordinate systems."""
    R_out = M_cam @ R @ M_world
    T_out = T @ M_cam
    return R_out, T_out


def pytorch3d_to_blender(
    R_w2cam_p3d: np.ndarray,
    T_w2cam_p3d: np.ndarray,
    world_y_up: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pytorch3d extrinsics to blender extrinsics."""
    M_world = _M_Y_UP_P3D_TO_BLENDER if world_y_up else np.eye(3, dtype=np.float32)
    return _convert_extrinsics(R_w2cam_p3d, T_w2cam_p3d, _M_CAM_FLIP, M_world)


def blender_to_pytorch3d(
    R_w2cam_blender: np.ndarray,
    T_w2cam_blender: np.ndarray,
    world_y_up: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert blender extrinsics to pytorch3d extrinsics."""
    M_world = _M_Y_UP_BLENDER_TO_P3D if world_y_up else np.eye(3, dtype=np.float32)
    return _convert_extrinsics(R_w2cam_blender, T_w2cam_blender, _M_CAM_FLIP, M_world)


def location_to_extrinsic(
    camera_dist: float,
    elevation_deg: float,
    azimuth_deg: float,
    blender_extrinsics: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 'look at view' extrinsics in blender or pytorch3d convention."""
    theta = math.radians(azimuth_deg)
    phi = math.radians(elevation_deg)

    # Compute camera position directly as numpy
    L_p3d = np.array(
        [
            [
                camera_dist * math.sin(phi) * math.cos(theta),
                camera_dist * math.cos(phi),
                -camera_dist * math.sin(phi) * math.sin(theta),
            ]
        ],
        dtype=np.float32,
    )

    # Compute pytorch3d extrinsics
    R_p3d = look_at_rotation(torch.from_numpy(L_p3d), up=((0, 1, 0),)).numpy()
    T_p3d = -L_p3d @ R_p3d
    R_p3d = R_p3d.transpose(0, 2, 1)

    return pytorch3d_to_blender(R_p3d, T_p3d) if blender_extrinsics else (R_p3d, T_p3d)


def extrinsics_to_cameras(
    tags: list[str],
    extrinsics: list[tuple[np.ndarray, np.ndarray]],
    camera_focal_length: float = 2.1875,
    fov: bool = False,
) -> dict[str, PerspectiveCameras]:
    """Convert a list of extrinsics to pytorch3d perspective cameras."""
    cameras = {}

    for tag, (R, T) in zip(tags, extrinsics):
        R_tensor = torch.tensor(R).T.unsqueeze(0)
        T_tensor = torch.tensor(T).unsqueeze(0)

        if fov:
            fov_degree = np.degrees(2 * np.arctan(1 / camera_focal_length))
            cameras[tag] = FoVPerspectiveCameras(fov=fov_degree, R=R_tensor, T=T_tensor)
        else:
            cameras[tag] = PerspectiveCameras(
                focal_length=torch.full((1, 2), camera_focal_length),
                principal_point=torch.zeros((1, 2)),
                R=R_tensor,
                T=T_tensor,
            )

    return cameras


def get_uniform_camera(
    distance: float = 12.0,
    elevation_deg: Optional[int] = None,
    n_cameras: int = 16,
    camera_focal_length: float = 2.1875,
    fov: bool = False,
) -> dict[str, PerspectiveCameras]:
    """Return uniformly spaced camera extrinsics."""
    elevation_cycle = cycle([elevation_deg] if elevation_deg else [70, 55, 85, 40])

    extrinsics = [
        tuple(
            np.squeeze(x)
            for x in location_to_extrinsic(
                distance, elev, (i / n_cameras) * 360, blender_extrinsics=False
            )
        )
        for i, elev in zip(range(n_cameras), elevation_cycle)
    ]

    return extrinsics_to_cameras(
        tags=[f"U{i:03d}" for i in range(n_cameras)],
        extrinsics=extrinsics,
        fov=fov,
        camera_focal_length=camera_focal_length,
    )
