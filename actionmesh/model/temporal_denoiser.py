# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from actionmesh.model.utils.block import FlowMatchingBlock
from actionmesh.model.utils.embeddings import scale_timestep
from actionmesh.model.utils.rotary_embedding import compute_rotary_embeddings
from actionmesh.model.utils.tensor_ops import merge_batch_time, split_batch_time
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from huggingface_hub import PyTorchModelHubMixin

DENOISER_SUBFOLDER = "denoiser"


@dataclass(eq=False)
class ActionMeshDenoiser(nn.Module, PyTorchModelHubMixin):
    """
    Temporal 3D denoiser (Stage I).
    Supports loading/saving via HuggingFace Hub.
    """

    # -- Nominal conditions
    num_tokens_nominal: int = 2048
    temporal_context_size: int = 16  # Number of frames model was trained to process

    # -- Denoiser/Flow-Matcher
    in_channels: int = 64
    num_layers: int = 21
    num_attention_heads: int = 16
    width: int = 2048
    mlp_ratio: float = 4.0
    cross_attention_dim: int = 1024

    # -- Inflate denoiser
    inflation_start: int = 0  # included
    inflation_end: int = -1  # excluded

    def __post_init__(self):
        super().__init__()
        self.out_channels = self.in_channels
        self.width_per_head = self.width // self.num_attention_heads
        self.latent_shape = (self.num_tokens_nominal, self.in_channels)

        # -- Time embedding
        self.time_embed = Timesteps(
            num_channels=self.width,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        # -- Linear projection
        self.time_proj = TimestepEmbedding(
            in_channels=self.width,
            time_embed_dim=self.width * 4,
            act_fn="gelu",
            out_dim=self.width,
        )

        # -- Denoiser
        self.proj_in = nn.Linear(self.in_channels, self.width, bias=True)

        def _should_inflate(layer: int) -> bool:
            """
            Decide if the block "i" should have inflated self-attention
            """
            if self.inflation_start != -1:
                if layer < self.inflation_start:
                    return False
            if self.inflation_end != -1:
                if layer >= self.inflation_end:
                    return False
            return True

        self.blocks = nn.ModuleList(
            [
                FlowMatchingBlock(
                    dim=self.width,
                    num_attention_heads=self.num_attention_heads,
                    use_self_attention=True,
                    inflate_self_attention=_should_inflate(layer),
                    use_cross_attention=True,
                    cross_attention_dim=self.cross_attention_dim,
                    cross_attention_norm_type=None,
                    attention_bias=False,
                    ff_activation="gelu",
                    ff_inner_dim=int(self.width * self.mlp_ratio),
                    skip=layer > self.num_layers // 2,
                )
                for layer in range(self.num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.out_channels, bias=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def precompute_freqs_rot(
        self,
        hidden_states: torch.Tensor,
        framestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Precompute rotary embeddings that remain constant across flow-matching denoising steps.

        Call this once before the denoising loop and pass the result to `forward`
        to avoid redundant computations.

        Args:
            hidden_states (B, T, N, D): Input latents (used for shape, dtype, and device).
            framestep (B, T): Video timesteps for temporal positional encoding.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (freqs_rot_cos, freqs_rot_sin) rotary embeddings.
        """
        _, _, N, _ = hidden_states.shape

        # -- Compute rotary positional encoding from framesteps
        framestep_rel = merge_batch_time(
            scale_timestep(
                framestep,
                center=True,
                scale=False,
            )
        )
        freqs_rot_cos, freqs_rot_sin = compute_rotary_embeddings(
            embed_dim=self.width_per_head,
            positions=framestep_rel,
        )
        freqs_rot_cos = freqs_rot_cos.unsqueeze(1).repeat(1, N + 1, 1).to(hidden_states)
        freqs_rot_sin = freqs_rot_sin.unsqueeze(1).repeat(1, N + 1, 1).to(hidden_states)

        return freqs_rot_cos, freqs_rot_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        framestep: torch.Tensor,
        diffusion_time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        freqs_rot: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform the denoising step of the flow-matching process.

        ActionMesh temporal 3D denoiser (stage I) synchronously denoises hidden_states using two conditioning sources:
            * Ground-truth latents (already in hidden_states, indicated by mask)
            * Context embeddings (image conditioning, used for cross-attention)

        The denoising is temporally aware via:
            1. Rotary positional embeddings derived from framestep
            2. Inflated self-attention across frames (in selected blocks)

        Shape legend:
            B: batch size
            T: number of frames
            N: number of tokens per frame
            D: latent dimension (in_channels)
            S: context sequence length

        Args:
            hidden_states (B, T, N, D): Input latents to denoise.
            context (B, T, S, cross_attention_dim): Context embeddings for cross-attention.
            framestep (B, T): Video timesteps for temporal positional encoding.
            diffusion_time (B): Flow-matching diffusion timestep (0=clean, 1=noise).
            mask (B, T): Optional mask for ground-truth latent positions.
                Values: 1=ground-truth (diffusion_time set to 0), 0=noise.
            freqs_rot: Optional precomputed rotary embeddings (freqs_rot_cos, freqs_rot_sin)
                from a previous forward call. Pass this to avoid recomputing in a denoising loop.

        Returns:
            tuple: (hidden_states, freqs_rot)
                - hidden_states (B, T, N, D): Denoised latents.
                - freqs_rot: Rotary embeddings tuple to pass to subsequent forward calls.
        """
        B, T, N, _ = hidden_states.shape

        # -- Use precomputed rotary embeddings if provided, otherwise compute them
        if freqs_rot is not None:
            freqs_rot_cos, freqs_rot_sin = freqs_rot
        else:
            freqs_rot_cos, freqs_rot_sin = self.precompute_freqs_rot(
                hidden_states, framestep
            )
            freqs_rot = (freqs_rot_cos, freqs_rot_sin)

        # -- Merge batch and frame dimensions
        hidden_states = merge_batch_time(hidden_states)  # (B, T, N, D) -> (B*T, N, D)
        hidden_states = self.proj_in(hidden_states)

        # -- Compute diffusion time embedding
        diffusion_time = diffusion_time.repeat(T)
        if mask is not None:
            # Zero out diffusion time for ground-truth frames (mask=1 means GT)
            diffusion_time = diffusion_time * (1 - merge_batch_time(mask))
        dt_emb = self.time_embed(diffusion_time).to(hidden_states.dtype)
        dt_emb = self.time_proj(dt_emb)

        # -- Prepend diffusion time token to sequence
        hidden_states = torch.cat([dt_emb.unsqueeze(dim=1), hidden_states], dim=1)

        # -- Transformer blocks with U-Net style skip connections
        skips = []
        context_merged = merge_batch_time(context)
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.num_layers // 2 else skips.pop()
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=context_merged,
                n_frames=T,
                freqs_rot=(freqs_rot_cos, freqs_rot_sin),
                skip=skip,
            )
            if layer < self.num_layers // 2:
                skips.append(hidden_states)

        # -- Output projection
        hidden_states = self.norm_out(hidden_states)
        # Remove prepended diffusion time token
        hidden_states = hidden_states[:, -N:]
        hidden_states = self.proj_out(hidden_states)

        # -- Restore original shape
        hidden_states = split_batch_time(
            hidden_states, T
        )  # (B*T, N, D) -> (B, T, N, D)

        return hidden_states, freqs_rot
