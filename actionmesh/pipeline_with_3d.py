# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import trimesh
from actionmesh.external.triposg import TripoSGVAE
from actionmesh.io.video_input import ActionMeshInput
from actionmesh.model.utils.storage import LatentBank, MeshBank
from actionmesh.pipeline import ActionMeshPipeline
from actionmesh.preprocessing.mesh_processor import (
    denormalize_mesh,
    merge_and_clean_mesh,
    NormalizationParams,
    normalize_mesh,
    sample_surface,
)

logger = logging.getLogger(__name__)


class ActionMeshPipelineWithMeshInput(ActionMeshPipeline):
    """ActionMesh pipeline variant for {video + 3D mesh} → 4D.

    Unlike the base :class:`ActionMeshPipeline` which generates the
    anchor 3D representation from a single frame via image-to-3D
    (TripoSG), this variant takes a user-provided anchor mesh and
    encodes it directly through the VAE.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae = None

    def _load_vae(self) -> None:
        """Load the image-to-3D pipeline to device. Does nothing if already loaded."""
        if self.vae is not None and self.vae.device == self.device:
            return
        if self.vae is None:
            logger.info("Loading VAE (TripoSG) model from disk...")
            self.vae = TripoSGVAE.from_pretrained(
                self._triposg_weights_dir,
                subfolder="vae",
            ).to(torch.float16)
        self.vae.to(self.device)

    def _load_all_models(self) -> None:
        """Load all models to target device (used when lazy_loading=False)."""
        self._load_background_removal()
        self._load_vae()
        self._load_image_encoder()
        self._load_temporal_denoiser()
        self._load_temporal_vae()

    def init_banks_from_anchor(
        self,
        input: ActionMeshInput,
        anchor_mesh: trimesh.Trimesh,
        seed: int = 44,
    ) -> tuple[
        LatentBank,
        MeshBank,
        NormalizationParams,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Generate anchor 3D representation from anchor frame using image-to-3D model.

        Uses TripoSG to generate an initial 3D mesh and latent from the anchor frame
        (specified by cfg.anchor_idx). The anchor latent and mesh serve as:
            1. Conditioning signal for Stage I (temporal 3D denoising)
            2. Reference topology for Stage II (all output meshes share anchor's faces)

        Args:
            input: ActionMeshInput containing input RGBA frames and video timesteps.
            anchor_mesh: Input mesh (modified in-place by merge + cleanup).
            seed: Random seed for 3D generation.

        Returns:
            latent_bank: LatentBank initialized with anchor latent.
            mesh_bank: MeshBank initialized with anchor mesh.
            normalization: Parameters used to normalize the mesh.
            vertex_merge_map: Mapping from original to merged vertex indices.
            pre_merge_faces: Original face array before merging.
        """
        vertex_merge_map, pre_merge_faces = merge_and_clean_mesh(
            anchor_mesh,
        )

        anchor_mesh, _params = normalize_mesh(anchor_mesh)
        surface = sample_surface(
            anchor_mesh,
            n_points=16384,
            seed=seed,
            with_normals=True,
            device=self.device,
            dtype=torch.float16,
        )
        anchor_latent = self.vae.encode_to_latent(surface)

        # -- Initialize empty banks
        latent_bank = LatentBank(
            verbose=True,
            empty_dims=self._denoiser_latent_shape,
        )
        mesh_bank = MeshBank(verbose=True)

        # -- Store anchor latent and mesh in banks
        anchor_timestep = input.timesteps[[self.cfg.anchor_idx]]
        latent_bank.update(timesteps=anchor_timestep, latents=anchor_latent)
        mesh_bank.update(meshes=[anchor_mesh], timesteps=anchor_timestep)

        return (
            latent_bank,
            mesh_bank,
            _params,
            vertex_merge_map,
            pre_merge_faces,
        )

    def __call__(
        self,
        input: ActionMeshInput,
        anchor_mesh: trimesh.Trimesh,
        seed: int = 44,
        # -- Optional parameter overrides
        stage_0_steps: int | None = None,
        face_decimation: int | None = None,
        floaters_threshold: float | None = None,
        stage_1_steps: int | None = None,
        guidance_scales: list[float] | None = None,
        anchor_idx: int | None = None,
    ) -> list[trimesh.Trimesh]:
        """Run the {video + 3D mesh} → 4D pipeline.

        Full pipeline execution:
            1. Preprocess input RGBA frames (grouped cropping & padding)
            2. Encode the provided anchor mesh into a 3D latent via the VAE
            3. Stage I: Denoise synchronized latents across all frames
                conditioned on (1) anchor 3D latent and (2) input frames
            4. Stage II: Decode latents into mesh deformations from the anchor mesh

        The returned meshes have their vertices expanded back to the
        original (pre-merge) topology so that UV / texture mapping
        is preserved.

        Args:
            input: ActionMeshInput containing input RGBA frames and timesteps.
            anchor_mesh: Reference mesh that defines the output topology; encoded via VAE to condition generation.
            seed: Random seed for generation.
            stage_0_steps: Number of inference steps for image-to-3D (TripoSG). Overrides config.
            face_decimation: Target number of faces for mesh decimation. Overrides config.
            floaters_threshold: Threshold for removing floaters (0.0-1.0). Overrides config.
            stage_1_steps: Number of flow-matching denoising steps in temporal 3D denoiser (stage I). Overrides config.
            guidance_scales: Classifier-free guidance scales in temporal 3D denoiser (stage I). Overrides config.
            anchor_idx: Index of the anchor frame (fixing the topology). Overrides config.

        Returns:
            Sequence of animated meshes (original topology)
            ordered by timestep.
        """
        if stage_0_steps is not None:
            self.cfg.model.image_to_3D_denoiser.num_inference_steps = stage_0_steps
        if stage_1_steps is not None:
            self.scheduler.num_inference_steps = stage_1_steps
        if guidance_scales is not None:
            self.cf_guidance.guidance_scales = guidance_scales
        if face_decimation is not None:
            self.mesh_process.face_decimation = face_decimation
        if floaters_threshold is not None:
            self.mesh_process.floaters_threshold = floaters_threshold
        if anchor_idx is not None:
            self.cfg.anchor_idx = anchor_idx

        # -- Preprocessing: remove background
        self._load_background_removal()
        input.frames = self.background_removal.process_images(input.frames)
        self._unload_model("background_removal")

        # -- Preprocessing: grouped cropping & padding
        input.frames = self.image_process.process_images(input.frames)

        with torch.inference_mode():

            # -- Stage 0: generate anchor latent from the input mesh
            self._load_vae()
            (
                latent_bank,
                mesh_bank,
                normalization,
                vertex_merge_map,
                pre_merge_faces,
            ) = self.init_banks_from_anchor(
                input,
                anchor_mesh,
                seed,
            )
            self._unload_model("vae")

            # -- Pre-compute context embeddings for all frames
            self._load_image_encoder()
            context = self.encode_all_frames(input)
            self._unload_model("image_encoder")

            # -- Stage I: denoise synchronized 3D latents
            self._load_temporal_denoiser()
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                latent_bank = self.generate_3d_latents(
                    input, context=context, latent_bank=latent_bank, seed=seed
                )
            self._unload_model("temporal_3D_denoiser")

            # -- Stage II: decode latents into mesh displacements
            self._load_temporal_vae()
            with torch.autocast(device_type="cuda", dtype=self._dtype):
                mesh_bank = self.generate_mesh_animation(
                    latent_bank=latent_bank, mesh_bank=mesh_bank
                )
            self._unload_model("temporal_3D_vae")

        meshes = mesh_bank.get_ordered(device="cpu")[0]

        # -- Postprocessing: unnormalize meshes and expand merged vertices back to original topology
        meshes = [denormalize_mesh(mesh, normalization) for mesh in meshes]
        meshes = [
            trimesh.Trimesh(
                vertices=m.vertices[vertex_merge_map],
                faces=pre_merge_faces,
                process=False,
            )
            for m in meshes
        ]

        return meshes
