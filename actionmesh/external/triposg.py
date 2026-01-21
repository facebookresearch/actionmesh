# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from triposg.pipelines.pipeline_triposg import TripoSGPipeline


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
            # Store latents on every step; final step will have the denoised latents
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
