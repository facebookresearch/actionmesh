# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import trimesh
from actionmesh.render.cameras import get_uniform_camera
from actionmesh.render.renderer import Renderer
from actionmesh.render.utils import resample_list, save_multiview_video_grid
from PIL import Image
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm

logger = logging.getLogger(__name__)


def array_to_img(X: torch.Tensor) -> Image.Image:
    if X.shape[1] == 1:
        X = X.repeat(1, 3, 1, 1)
    X = X.permute(0, 2, 3, 1).cpu().numpy()
    X = (X * 255).astype(np.uint8)
    return Image.fromarray(X[0])


def image_and_mask_to_rgba(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Return RGBA image from RGB+Mask(L).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)
    rgba_image = image.copy()
    rgba_image.putalpha(mask)
    return rgba_image


def trimesh_to_pytorch3d(mesh: trimesh.Trimesh, device: str = "cpu") -> Meshes:
    """Convert a trimesh mesh to a pytorch3d Meshes object."""
    verts = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).long().to(device)
    # -- Blue texture
    blue_color = torch.tensor([0.0, 0.0, 1.0]).to(device)
    vertex_colors = [blue_color.expand(verts.shape[0], -1)]
    texture = TexturesVertex(vertex_colors)

    return Meshes(verts=[verts], faces=[faces], textures=texture)


class ActionMeshVisualizer(nn.Module):
    """Visualizer to render from pipeline output"""

    def __init__(
        self,
        image_size: int = 256,
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cameras: list[str] = ["U000", "U004", "U008"],
    ):
        super().__init__()
        self.image_size = image_size

        # -- Initialize renderer
        self.renderer = Renderer(
            img_size=image_size,
            background_color=bg_color,
        )

        # -- Initialize cameras
        self.cameras = {
            k: v for (k, v) in get_uniform_camera(distance=3.0).items() if k in cameras
        }

    @torch.no_grad()
    def render(
        self,
        meshes: list[trimesh.Trimesh],
        device: str,
        output_dir: str,
        input_frames: Optional[list[Image.Image]],
    ):
        """Render modalities from an output"""

        n_cameras = len(self.cameras)
        n_frames = len(meshes)

        # If input frames are given, adapt to the number of meshes
        if input_frames is not None:
            input_frames = resample_list(input_frames, n_frames)

        self.renderer = self.renderer.to(device)
        global_predictions = []

        if input_frames is not None:
            global_predictions.append(
                [{"mask": frame, "normal": frame} for frame in input_frames]
            )

        pbar = tqdm(total=n_cameras * n_frames, desc="Rendering")
        for view_id, camera in self.cameras.items():
            predictions = []
            for mesh in meshes:
                pbar.set_postfix(
                    camera=view_id, frame=f"{len(predictions)+1}/{n_frames}"
                )

                mesh_p3d = trimesh_to_pytorch3d(mesh, device=device)
                camera = camera.to(device)

                (
                    image_render,
                    mask_render,
                    normal_render,
                ) = self.renderer(
                    meshes=mesh_p3d,
                    cameras=camera,
                    return_normals=True,
                )

                preds = {
                    "mask": array_to_img(mask_render),
                    "normal": array_to_img(normal_render),
                }

                preds = {
                    k: image_and_mask_to_rgba(preds[k], preds["mask"]) for k in preds
                }
                predictions.append(preds)
                pbar.update(1)

            global_predictions.append(predictions)
        pbar.close()

        saved_files = save_multiview_video_grid(
            global_predictions,
            output_dir,
            modalities=["normal"],
            n_cols=n_cameras + 1 if input_frames is not None else n_cameras,
            image_size=self.image_size,
        )

        for filepath in saved_files:
            logger.info(f"Saved render: {filepath}")
