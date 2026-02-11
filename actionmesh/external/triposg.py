# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from triposg.inference_utils import hierarchical_extract_geometry
from triposg.models.autoencoders import TripoSGVAEModel
from triposg.pipelines.pipeline_triposg import TripoSGPipeline

try:
    from actionmesh.model.utils.pointcloud_sampling import sample_pc, sample_pc_grouped
    from pytorch3d.ops.utils import masked_gather

    _is_pytorch3d_available = True
except ImportError:
    _is_pytorch3d_available = False


class TripoSGPipelinePlus(TripoSGPipeline):
    """
    Extended TripoSG pipeline that returns both latents and mesh.

    Inherits all functionality from TripoSGPipeline and captures the final
    latents via callback mechanism.
    """

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, trimesh.Trimesh]:
        """
        Generate 3D mesh from image, returning both latents and mesh.

        Returns:
            Tuple of (latents, mesh): The denoised latent and extracted trimesh.
        """
        # Storage for capturing final latents via callback
        captured_latents = {}

        def capture_latents_callback(_pipe, _step, _timestep, callback_kwargs):
            captured_latents["latents"] = callback_kwargs["latents"].clone()
            return callback_kwargs

        # Call parent pipeline with callback to capture latents
        output = super().__call__(
            image=image,
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_shapes_per_prompt=num_shapes_per_prompt,
            generator=generator,
            latents=latents,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=capture_latents_callback,
            callback_on_step_end_tensor_inputs=["latents"],
            bounds=bounds,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
            flash_octree_depth=flash_octree_depth,
            use_flash_decoder=use_flash_decoder,
            return_dict=return_dict,
        )

        # Extract mesh from output
        if return_dict:
            mesh = output.meshes[0]
        else:
            mesh = output[1][0]

        return captured_latents["latents"], mesh


class TripoSGVAE(TripoSGVAEModel):
    """Extended TripoSG VAE with a convenience latent-sampling helper."""

    def __init__(self, *args, **kwargs):
        if not _is_pytorch3d_available:
            raise ImportError(
                "pytorch3d is required for TripoSGVAE but is not installed. "
            )
        super().__init__(*args, **kwargs)

    def _sample_features(
        self,
        x: torch.Tensor,
        num_tokens: int,
        seed: Optional[int] = None,
        grouped_fps_n: Optional[int] = None,
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        indices = np.random.default_rng(seed).choice(
            x.shape[1],
            num_tokens * 4,
            replace=num_tokens * 4 > x.shape[1],
        )
        selected_points = x[:, indices]

        if grouped_fps_n is not None:
            _, sampled_indices = sample_pc_grouped(
                points=selected_points[..., :3],
                n_samples=num_tokens,
                n_grouped_frames=grouped_fps_n,
                sampling_type="fps",
                fps_random=True,
            )
        else:
            _, sampled_indices = sample_pc(
                points=selected_points[..., :3],
                n_samples=num_tokens,
                sampling_type="fps",
                fps_random=True,
            )

        return masked_gather(selected_points, sampled_indices)

    @torch.no_grad()
    def encode_to_latent(
        self,
        surface: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a surface tensor into a latent sample.

        Args:
            surface: Tensor of shape ``[B, N, 6]``
                (xyz + normals).

        Returns:
            Latent sample tensor from the VAE posterior.
        """
        surface = surface.to(
            self.device,
            dtype=self.dtype,
        )
        posterior = self.encode(surface).latent_dist
        return posterior.sample()

    @torch.no_grad()
    def decode_latents(
        self,
        latents: torch.Tensor,
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
    ) -> list[trimesh.Trimesh]:
        """Decode latents into meshes."""

        geometric_func = lambda x: self.decode(latents, sampled_points=x).sample

        output = hierarchical_extract_geometry(
            geometric_func,
            self.device,
            bounds=bounds,
            dense_octree_depth=dense_octree_depth,
            hierarchical_octree_depth=hierarchical_octree_depth,
        )
        meshes = [
            trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1])
            for mesh_v_f in output
        ]

        return meshes
