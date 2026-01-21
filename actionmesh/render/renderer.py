# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)
import torch.nn.functional as Fu
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    FoVPerspectiveCameras,
    HardPhongShader,
    Materials,
    MeshRasterizer,
    MeshRendererWithFragments,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import nn
from torch.nn import functional as F

EPS = 1e-8
MAX_N_FACES_OPENGL = 25_000_000

try:
    from pytorch3d.renderer.opengl import MeshRasterizerOpenGL

    _OPENGL_BACKEND = True
except ImportError:
    logger.warning("opengl backend not found")
    _OPENGL_BACKEND = False


class Renderer(nn.Module):
    def __init__(self, img_size: int, **kwargs):
        super().__init__()

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.opengl_backend = kwargs.pop("opengl_backend", False)

        self.lights = AmbientLights()

        r_kwargs = {
            "perspective_correct": kwargs.pop("perspective_correct", None),
            "clip_barycentric_coords": True,
            "z_clip_value": kwargs.pop("z_clip", None),
        }
        raster_settings = RasterizationSettings(
            image_size=(self.img_size[0] * 2, self.img_size[1] * 2),
            bin_size=0,  # rendering bugs for large number of faces with default
            # see https://github.com/facebookresearch/pytorch3d/issues/348
            max_faces_opengl=MAX_N_FACES_OPENGL,
            **r_kwargs,
        )

        if self.opengl_backend and _OPENGL_BACKEND:
            rasterizer_cls = MeshRasterizerOpenGL
        else:
            rasterizer_cls = MeshRasterizer

        s_kwargs = {"cameras": None, "lights": self.lights}
        s_kwargs["blend_params"] = BlendParams(
            background_color=kwargs.pop("background_color", (0, 0, 0)),
            sigma=0,
        )
        shader_cls = HardPhongShaderPlus

        self.renderer = VizMeshRendererWithFragments(
            rasterizer_cls(cameras=None, raster_settings=raster_settings),
            shader_cls(**s_kwargs),
        )

    def to(self, device):
        super().to(device)
        self.renderer = self.renderer.to(device)
        return self

    def forward(
        self,
        meshes: Meshes,
        cameras: PerspectiveCameras,
        return_normals: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        # XXX bug when using perspective_correct, see https://github.com/facebookresearch/pytorch3d/issues/561
        # setting eps fixes the issue
        images, fragments = self.renderer(meshes, eps=EPS, cameras=cameras)
        images, masks = images.split([3, 1], dim=1)

        normal = None
        if return_normals:
            normal = soft_normal_shading(
                fragments=fragments,
                meshes=meshes,
                cameras=cameras,
            )

        # -- Post-processing
        masks = masks.clamp(0.0, 1.0)
        if return_normals:
            normal = normal.clamp(0.0, 1.0)
            normal = make_normal_image(masks, normal)

        return images, masks, normal  # BCHW


class VizMeshRendererWithFragments(MeshRendererWithFragments):
    """Renderer for visualization, with anti-aliasing"""

    @torch.no_grad()
    def __call__(self, *input, **kwargs):
        images, fragments = super().__call__(*input, **kwargs)
        return F.avg_pool2d(images, kernel_size=2, stride=2), fragments


class HardPhongShaderPlus(HardPhongShader):
    """Rewriting to permute output tensor + working `to` method for multi-gpus"""

    def forward(self, fragments, meshes, **kwargs):
        return super().forward(fragments, meshes, **kwargs).permute(0, 3, 1, 2)

    def to(self, device):
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self


def soft_normal_shading(
    fragments,
    meshes: Meshes,
    cameras: PerspectiveCameras,
    rescale: bool = True,
    **kwargs,
):
    """Soft normal shading using interpolated vertex normals for smooth appearance."""
    cameras_sphere = copy.copy(cameras)
    cameras_sphere.T = cameras_sphere.T / 2.0

    # Use interpolated vertex normals instead of face normals
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]  # (F, 3, 3)
    # Interpolate normals using barycentric coordinates
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    shape = pixel_normals.shape
    pixel_normals = cameras_sphere.get_world_to_view_transform(
        **kwargs
    ).transform_points(pixel_normals.view(shape[0], -1, 3), eps=kwargs.get("eps", 1e-7))
    pixel_normals = pixel_normals.view(*shape)
    if rescale:
        pixel_normals = (F.normalize(pixel_normals, dim=-1) + 1) * 0.5
    # (N, H, W, K, 3) -> (N, 3, H, W) - take first face per pixel
    pixel_normals = pixel_normals[..., 0, :].permute((0, 3, 1, 2))
    return pixel_normals


def make_normal_image(mask_render: torch.Tensor, normal_render: torch.Tensor):

    mask_cpu = mask_render.cpu().detach().clone()
    normal_cpu = normal_render.cpu().detach().clone()

    normal_resize = Fu.interpolate(
        normal_cpu,
        size=mask_cpu.shape[2:],
        mode="nearest",
    )
    normal_cpu = normal_resize * mask_cpu + (1 - mask_cpu)
    return normal_cpu
