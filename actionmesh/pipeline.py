# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import trimesh
from actionmesh.external.triposg import TripoSGPipelinePlus
from actionmesh.io.video_input import ActionMeshInput
from actionmesh.model.image_encoder import ImageEncoder
from actionmesh.model.temporal_autoencoder import (
    ActionMeshAutoencoder,
    AUTOENCODER_SUBFOLDER,
)
from actionmesh.model.temporal_denoiser import ActionMeshDenoiser, DENOISER_SUBFOLDER
from actionmesh.model.utils.embeddings import interpolate_timesteps, scale_timestep
from actionmesh.model.utils.storage import LatentBank, MeshBank
from actionmesh.model.utils.timesteps import chunk_from
from actionmesh.preprocessing.background_removal import BackgroundRemover
from actionmesh.preprocessing.image_processor import ImagePreprocessor
from actionmesh.preprocessing.mesh_processor import get_mesh_features, MeshPostprocessor
from actionmesh.utils import download_if_missing, force_memory_cleanup, load_config
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class ActionMeshPipeline(nn.Module):
    """
    ActionMesh pipeline for video-to-4D mesh generation.

    This pipeline combines:
        - Off-the-shelf image-to-3D model (3D output returned as latent+mesh)
        - Stage I (temporal_3D_denoiser): Flow-matching denoiser that generates synchronized 3D latents
        - Stage II (temporal_3D_vae): Autoencoder that decodes 3D latents into mesh displacements
    """

    def __init__(
        self,
        config_name: str,
        config_dir: str,
        dtype: torch.dtype = torch.bfloat16,
        lazy_loading: bool = False,
    ):
        """
        Initialize the ActionMesh pipeline.

        Args:
            config_name: Name of the config file (e.g., "actionmesh.yaml").
            config_dir: Path to the config directory.
            dtype: Data type for mixed precision inference (torch.float16 or torch.bfloat16).
            lazy_loading: If True (default), models are loaded on-demand and unloaded
                after use to minimize memory. If False, all models are loaded once
                when to() is called and stay on GPU.
        """
        super().__init__()
        self.cfg = load_config(config_name, config_dir)

        logger.info("Downloading external models if missing...")
        # -- Download external HuggingFace checkpoints (store paths for lazy loading)
        self._triposg_weights_dir = "pretrained_weights/TripoSG"
        download_if_missing(
            repo_id="VAST-AI/TripoSG", local_dir=self._triposg_weights_dir
        )
        self._dinov2_weights_dir = "pretrained_weights/dinov2"
        download_if_missing(
            repo_id="facebook/dinov2-large", local_dir=self._dinov2_weights_dir
        )
        self._rmbg_weights_dir = "pretrained_weights/RMBG"
        download_if_missing(repo_id="briaai/RMBG-1.4", local_dir=self._rmbg_weights_dir)
        self._actionmesh_weights_dir = "pretrained_weights/ActionMesh"
        download_if_missing(
            repo_id="facebook/ActionMesh",
            local_dir=self._actionmesh_weights_dir,
        )
        logger.info("Downloading external models if missing... Done")

        # Initialize model references to None (lazy loading - models loaded on demand)
        self.image_to_3d_pipe: Optional[TripoSGPipelinePlus] = None
        self.background_removal: Optional[BackgroundRemover] = None
        self.temporal_3D_denoiser: Optional[ActionMeshDenoiser] = None
        self.image_encoder: Optional[ImageEncoder] = None
        self.temporal_3D_vae: Optional[ActionMeshAutoencoder] = None

        self._denoiser_latent_shape = tuple(self.cfg.denoiser_latent_shape)

        # -- Load lightweight components (always loaded)
        self.image_process = ImagePreprocessor()
        self.mesh_process: MeshPostprocessor = instantiate(
            self.cfg.model.mesh_process,
            _convert_="partial",
        )()
        # -- Load scheduler and classifier-free guidance
        self.scheduler = instantiate(
            self.cfg.model.scheduler,
            _convert_="partial",
        )()
        self.cf_guidance = instantiate(
            self.cfg.model.cf_guidance,
            _convert_="partial",
        )()

        # Initialize target device and dtype (models stay on CPU until needed)
        self._target_device = torch.device("cpu")
        self._dtype = dtype
        self._lazy_loading = lazy_loading

    def _unload_model(self, model_attr: str) -> None:
        """Completely unload a model from memory to free system RAM.
        Args:
            model_attr: Attribute name of the model to unload (e.g., 'image_to_3d_pipe')
        """
        if not self._lazy_loading:
            return
        model = getattr(self, model_attr, None)
        if model is not None:
            del model
            setattr(self, model_attr, None)
        force_memory_cleanup()
        logger.debug(f"Unloaded model: {model_attr}")

    def _load_image_to_3d(self) -> None:
        """Load the image-to-3D pipeline to device. Does nothing if already loaded."""
        if (
            self.image_to_3d_pipe is not None
            and self.image_to_3d_pipe.device == self.device
        ):
            return
        if self.image_to_3d_pipe is None:
            logger.info("Loading image-to-3D (TripoSG) model from disk...")
            self.image_to_3d_pipe = TripoSGPipelinePlus.from_pretrained(
                self._triposg_weights_dir
            ).to(torch.float16)
        self.image_to_3d_pipe.to(self.device)

    def _load_background_removal(self) -> None:
        """Load the background removal model to device. Does nothing if already loaded."""
        if (
            self.background_removal is not None
            and self.background_removal.device == self.device
        ):
            return
        if self.background_removal is None:
            logger.info("Loading background removal model from disk...")
            self.background_removal = BackgroundRemover(self._rmbg_weights_dir)
            self.background_removal.eval()
        self.background_removal.to(self.device)

    def _load_image_encoder(self) -> None:
        """Load the image encoder model to device. Does nothing if already loaded."""
        if self.image_encoder is not None and self.image_encoder.device == self.device:
            return
        if self.image_encoder is None:
            logger.info("Loading image encoder from disk...")
            self.image_encoder = instantiate(
                self.cfg.model.image_encoder,
                _convert_="partial",
            )()
            self.image_encoder.eval()
        self.image_encoder.to(self.device)

    def _load_temporal_denoiser(self) -> None:
        """Load the temporal 3D denoiser to device. Does nothing if already loaded."""
        if (
            self.temporal_3D_denoiser is not None
            and self.temporal_3D_denoiser.device == self.device
        ):
            return
        if self.temporal_3D_denoiser is None:
            logger.info("Loading temporal 3D denoiser (Stage I) from disk...")
            self.temporal_3D_denoiser = ActionMeshDenoiser.from_pretrained(
                f"{self._actionmesh_weights_dir}/{DENOISER_SUBFOLDER}"
            )
            self.temporal_3D_denoiser.eval()
        self.temporal_3D_denoiser.to(self.device)

    def _load_temporal_vae(self) -> None:
        """Load the temporal 3D VAE to device. Does nothing if already loaded."""
        if (
            self.temporal_3D_vae is not None
            and self.temporal_3D_vae.device == self.device
        ):
            return
        if self.temporal_3D_vae is None:
            logger.info("Loading temporal 3D VAE (Stage II) from disk...")
            self.temporal_3D_vae = ActionMeshAutoencoder.from_pretrained(
                f"{self._actionmesh_weights_dir}/{AUTOENCODER_SUBFOLDER}"
            )
            self.temporal_3D_vae.eval()
        self.temporal_3D_vae.to(self.device)

    @property
    def device(self) -> torch.device:
        return self._target_device

    def to(self, device: str) -> "ActionMeshPipeline":
        """
        Set the target device for inference.

        If lazy_loading=True (default), models stay on CPU and are loaded
        on-demand during __call__ to minimize peak memory.
        If lazy_loading=False, all models are loaded immediately and kept on GPU.

        Args:
            device: Target device (e.g., "cuda", "cuda:0", "cpu")

        Returns:
            self for method chaining
        """
        self._target_device = torch.device(device)
        if not self._lazy_loading:
            self._load_all_models()
        return self

    def _load_all_models(self) -> None:
        """Load all models to target device (used when lazy_loading=False)."""
        self._load_background_removal()
        self._load_image_to_3d()
        self._load_image_encoder()
        self._load_temporal_denoiser()
        self._load_temporal_vae()

    def encode_all_frames(
        self,
        input: ActionMeshInput,
    ) -> torch.Tensor:
        """
        Pre-compute context embeddings for all input frames.

        Args:
            input: ActionMeshInput containing all video frames.

        Returns:
            context (T, S, D): Context embeddings for all T frames.
        """
        return self.image_encoder.encode_images(input.frames)

    def _denoise_latents(
        self,
        input: ActionMeshInput,
        context: torch.Tensor,
        latent_bank: LatentBank,
        seed: int = 44,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """
        Denoise latents for a single AR window via flow-matching.

        Called by generate_3d_latents() for each autoregressive window.
        Starting from noise, iteratively denoises to produce T synchronized
        3D latents conditioned on:
            1. Pre-computed context embeddings
            2. Previously computed latents from latent_bank

        Shape legend:
            B: batch size (1)
            T: number of video timesteps in this AR window
            N: number of 3D tokens per frame
            D: latent embedding dimension

        Args:
            input: ActionMeshInput for this AR window (T frames with timesteps).
            context: Pre-computed context embeddings for this window (T, S, D).
            latent_bank: LatentBank containing previously computed 3D latents.
            seed: Random seed for noise initialization.
            step_callback: Optional callback at each step with (step, total_steps).

        Returns:
            latents (B, T, N, D): Denoised 3D latents for all T timesteps.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # -- Retrieve conditioning latents (already computed) and mask from bank
        cond_latents, cond_mask = latent_bank.get(
            timesteps=input.timesteps, device=self.device, add_batch_dim=True
        )

        # -- Sample initial noise for timesteps not in bank
        init_noise = self.scheduler.get_noise(
            batch_size=1,
            latent_shape=self._denoiser_latent_shape,
            n_timesteps=input.n_frames,
            generator=generator,
            device=self.device,
        )

        # -- Initialize latents: conditioning where available, noise elsewhere
        init_latent = cond_latents * cond_mask[..., None, None] + init_noise * (
            1.0 - cond_mask[..., None, None]
        )

        # -- Iterative flow-matching denoising
        latents = self.scheduler.denoise(
            self.temporal_3D_denoiser,
            self.cf_guidance,
            init_latent=init_latent,
            context=context[None],
            mask=cond_mask.to(init_latent.dtype),
            framestep=input.timesteps[None],
            device=self.device,
            disable_prog=False,
            step_callback=step_callback,
        )

        return latents

    def _decode_displacement(
        self,
        latents: torch.Tensor,
        window_timesteps: torch.Tensor,
        source_alpha: torch.Tensor,
        target_alphas: torch.Tensor,
        anchor_mesh: trimesh.Trimesh,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[trimesh.Trimesh]:
        """
        Decode 3D latents into mesh displacement fields for a single AR window.

        Called by generate_mesh_animation() for each autoregressive window. Given denoised
        3D latents and an anchor mesh, predicts per-vertex displacement fields to produce
        a sequence of deformed meshes.

        Shape legend:
            B: batch size (1)
            T: number of video timesteps in this AR window
            N: number of 3D tokens per frame
            D: latent embedding dimension
            T_out: number of output timesteps (after temporal subsampling)
            V: number of mesh vertices

        Args:
            latents (B, T, N, D): Denoised 3D latents from Stage I for this window.
            window_timesteps (B, T): Video timesteps for temporal positional encoding.
            source_alpha (B,): Anchor timestep in normalized time [0, 1].
            target_alphas (B, T_out): Target timesteps in normalized time [0, 1].
            anchor_mesh: Anchor mesh whose vertices will be displaced.
            step_callback: Optional callback called at each step with (step, total_steps).

        Returns:
            list[trimesh.Trimesh]: Sequence of T_out deformed meshes sharing anchor topology.
        """
        _, n_output_timesteps = target_alphas.shape

        # -- Extract vertex positions and normals from anchor mesh
        vertex_features = get_mesh_features(anchor_mesh, with_normals=True)[None].to(
            self.device
        )

        # -- Predict per-vertex displacement fields via temporal 3D autoencoder
        displacement = self.temporal_3D_vae(
            latent=latents,
            framestep=window_timesteps,
            source_alpha=source_alpha,
            target_alphas=target_alphas,
            query=vertex_features,
            step_callback=step_callback,
        )

        # -- Apply displacement to anchor vertices
        deformed_vertices = self.temporal_3D_vae.apply_displacement(
            vertex=vertex_features[:3],
            displacement=displacement,
        )
        deformed_vertices_np = deformed_vertices.cpu().numpy()

        # -- Build output meshes (all share anchor topology)
        output_meshes = [
            trimesh.Trimesh(
                vertices=deformed_vertices_np[0, i],
                faces=anchor_mesh.faces,
                process=False,
            )
            for i in range(n_output_timesteps)
        ]

        return output_meshes

    def init_banks_from_anchor(
        self,
        input: ActionMeshInput,
        seed: int = 44,
    ) -> tuple[LatentBank, MeshBank]:
        """
        Generate anchor 3D representation from anchor frame using image-to-3D model.

        Uses TripoSG to generate an initial 3D mesh and latent from the anchor frame
        (specified by cfg.anchor_idx). The anchor latent and mesh serve as:
            1. Conditioning signal for Stage I (temporal 3D denoising)
            2. Reference topology for Stage II (all output meshes share anchor's faces)

        Args:
            input: ActionMeshInput containing input RGBA frames and video timesteps.
            seed: Random seed for 3D generation.

        Returns:
            latent_bank: LatentBank initialized with anchor latent.
            mesh_bank: MeshBank initialized with anchor mesh.
        """
        # -- Generate 3D mesh and latent from anchor frame via TripoSG
        anchor_latent, anchor_mesh = self.image_to_3d_pipe(
            image=input.frames[self.cfg.anchor_idx],
            generator=torch.Generator(device=self.image_to_3d_pipe.device).manual_seed(
                seed
            ),
            num_inference_steps=self.cfg.model.image_to_3D_denoiser.num_inference_steps,
            guidance_scale=self.cfg.model.image_to_3D_denoiser.guidance_scale,
        )

        # -- Post-process mesh (decimate, remove floaters)
        anchor_mesh = self.mesh_process.process_mesh(anchor_mesh, seed=seed)

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

        return latent_bank, mesh_bank

    def generate_3d_latents(
        self,
        input: ActionMeshInput,
        context: torch.Tensor,
        latent_bank: LatentBank,
        seed: int = 44,
        step_callback: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> LatentBank:
        """
        Stage I: Generate synchronized 3D latents for all video frames.

        For sequences longer than the model's temporal context (16 frames), uses an autoregressive
        sliding window approach:
            1. Anchor frame's latent is already in latent_bank (from init_banks_from_anchor)
            2. Process overlapping windows of frames, conditioning on previously computed latents
            3. Slide window forward, using overlap for temporal coherence

        Example for 31 frames with model_window=16, slide=15, anchor=0:
            Window 1: frames [0-15], condition on [0], denoise [1-15]
            Window 2: frames [15-31], condition on [15], denoise [16-31]

        Args:
            input: ActionMeshInput containing input RGBA frames and video timesteps.
            context: Pre-computed context embeddings for all frames (T, S, D).
            latent_bank: LatentBank pre-initialized with anchor latent (from init_banks_from_anchor).
                Updated in-place as each window is processed.
            seed: Random seed for noise initialization.
            step_callback: Optional callback called at each denoising step with
                (step, total_steps, window_idx, total_windows).

        Returns:
            latent_bank: LatentBank containing denoised 3D latents for all input timesteps.
        """

        # -- Partition timesteps into overlapping windows for autoregressive (AR) denoising
        ar_windows = chunk_from(
            start=self.cfg.anchor_idx,
            total=input.n_frames,
            size=self.cfg.model.temporal_3D_denoiser.temporal_context_size,
            slide=self.cfg.sliding_window_denoiser,
        )

        total_windows = len(ar_windows)
        for i, window_indices in enumerate(ar_windows):
            # -- Extract input (frames + timesteps) for this AR window
            window_input = input.get(window_indices)
            # -- Extract pre-computed context for this window
            window_context = context[window_indices]

            # -- [Optional] Create step callback with window info
            _step_cb = None
            if step_callback is not None:

                def _step_cb(
                    step: int, total: int, _i: int = i, _tw: int = total_windows
                ) -> None:
                    step_callback(step, total, _i, _tw)

            # -- Flow-matching denoising conditioned on latent_bank and context
            window_latents = self._denoise_latents(
                input=window_input,
                context=window_context,
                latent_bank=latent_bank,
                seed=seed + i,
                step_callback=_step_cb,
            )

            # -- Store denoised latents for conditioning in subsequent AR steps
            latent_bank.update(
                latents=window_latents,
                timesteps=window_input.timesteps,
            )

        return latent_bank

    def generate_mesh_animation(
        self,
        latent_bank: LatentBank,
        mesh_bank: MeshBank,
        step_callback: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> MeshBank:
        """
        Stage II: Generate animated mesh sequence by decoding 3D latents into displacement fields.

        For sequences longer than the model's temporal context (16 frames), uses an autoregressive
        sliding window approach similar to Stage I:
            1. Anchor mesh is already in mesh_bank (from init_banks_from_anchor)
            2. Process overlapping windows of latents, predicting displacement from anchor to each frame
            3. Slide window forward, reusing anchor mesh topology throughout

        Args:
            latent_bank: LatentBank containing denoised 3D latents from generate_3d_latents.
            mesh_bank: MeshBank pre-initialized with anchor mesh (from init_banks_from_anchor).
                Updated in-place as each window is processed.
            step_callback: Optional callback called at each decoding step with
                (step, total_steps, window_idx, total_windows).

        Returns:
            mesh_bank: MeshBank containing deformed meshes for all output timesteps.
                All meshes share identical topology (faces) with the anchor mesh.
        """

        # -- Partition timesteps into overlapping windows for autoregressive (AR) decoding
        ar_windows = chunk_from(
            start=self.cfg.anchor_idx,
            total=latent_bank.n_timesteps,
            size=self.cfg.model.temporal_3D_vae.temporal_context_size,
            slide=self.cfg.sliding_window_autoencoder,
        )

        all_timesteps = latent_bank.get_ordered_timesteps().to(self.device)
        total_windows = len(ar_windows)
        for window_idx, window_indices in enumerate(ar_windows):
            window_timesteps = all_timesteps[window_indices][None]

            # -- Retrieve denoised latents for this window from Stage I
            window_latents, _ = latent_bank.get(
                timesteps=window_timesteps[0], device=self.device, add_batch_dim=True
            )

            # -- Get anchor mesh (source for displacement prediction)
            anchor_mesh = mesh_bank.get(timesteps=window_timesteps[:, 0])[0]
            assert anchor_mesh is not None, "Anchor mesh should be in mesh_bank"

            # -- Interpolate timesteps for temporal subsampling (higher output resolution)
            output_timesteps = interpolate_timesteps(
                window_timesteps,
                subsampling_level=self.cfg.subsampling_level,
                device=self.device,
                drop_first=True,
            )

            # -- Normalize timesteps to [0, 1] for displacement field prediction
            source_alpha = scale_timestep(window_timesteps, center=True, scale=True)[
                :, 0
            ]
            target_alphas = scale_timestep(output_timesteps, center=True, scale=True)

            # -- [Optional] Create step callback with window info
            _step_cb = None
            if step_callback is not None:

                def _step_cb(
                    step: int,
                    total: int,
                    _i: int = window_idx,
                    _tw: int = total_windows,
                ) -> None:
                    step_callback(step, total, _i, _tw)

            # -- Decode latents into per-vertex displacement fields
            window_meshes = self._decode_displacement(
                latents=window_latents,
                window_timesteps=window_timesteps,
                source_alpha=source_alpha,
                target_alphas=target_alphas,
                anchor_mesh=anchor_mesh,
                step_callback=_step_cb,
            )

            # -- Store deformed meshes for this window
            mesh_bank.update(
                meshes=window_meshes,
                timesteps=output_timesteps[0],
            )

        return mesh_bank

    def __call__(
        self,
        input: ActionMeshInput,
        seed: int = 44,
        # -- Optional parameter overrides
        stage_0_steps: int | None = None,
        face_decimation: int | None = None,
        floaters_threshold: float | None = None,
        stage_1_steps: int | None = None,
        guidance_scales: list[float] | None = None,
        anchor_idx: int | None = None,
    ) -> list[trimesh.Trimesh]:
        """
        Run the video → 4D pipeline.

        Full pipeline execution:
            1. Preprocess input RGBA frames (grouped cropping & padding)
            2. Generate an anchor 3D mesh & latent from a single frame (TripoSG)
            3. Stage I: Denoise synchronized latents across all frames
                conditioned on (1) anchor 3D latent and (2) input frames
            4. Stage II: Decode latents into mesh deformations from the anchor mesh

        Args:
            input: ActionMeshInput containing input RGBA frames and timesteps.
            seed: Random seed for generation.
            stage_0_steps: Number of inference steps for image-to-3D (TripoSG). Overrides config.
            face_decimation: Target number of faces for mesh decimation. Overrides config.
            floaters_threshold: Threshold for removing floaters (0.0-1.0). Overrides config.
            stage_1_steps: Number of flow-matching denoising steps in temporal 3D denoiser (stage I). Overrides config.
            guidance_scales: Classifier-free guidance scales in temporal 3D denoiser (stage I). Overrides config.
            anchor_idx: Index of the anchor frame (fixing the topology). Overrides config.

        Returns:
            list[trimesh.Trimesh]: Sequence of animated meshes (fixed topology) ordered by timestep.
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
            # -- Stage 0: generate anchor 3D mesh & latent from single frame
            self._load_image_to_3d()
            latent_bank, mesh_bank = self.init_banks_from_anchor(input, seed)
            self._unload_model("image_to_3d_pipe")

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

        return mesh_bank.get_ordered(device="cpu")[0]
