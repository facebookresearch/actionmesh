# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Deformation head -- Stage II

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from actionmesh.model.utils.block import FlowMatchingBlock
from actionmesh.model.utils.embeddings import (
    FrequencyPositionalEmbedding,
    scale_timestep,
    TimestepEmbedder,
)
from actionmesh.model.utils.rotary_embedding import compute_rotary_embeddings
from actionmesh.model.utils.tensor_ops import merge_batch_time, merge_time_tokens
from huggingface_hub import PyTorchModelHubMixin
from tqdm import tqdm

logger = logging.getLogger(__name__)

AUTOENCODER_SUBFOLDER = "autoencoder"


@dataclass(eq=False)
class ActionMeshAutoencoder(nn.Module, PyTorchModelHubMixin):
    """
    Temporal 3D autoencoder (Stage II).
    Supports loading/saving via HuggingFace Hub
    """

    verbose: bool = True

    # -- Nominal conditions
    temporal_context_size: int = 16  # Number of frames model was trained to process

    # -- Decoder
    in_channels: int = 3
    in_extra_channels: int = 3
    out_dim: int = 3
    latent_channels: int = 64
    width: int = 1024
    num_layers: int = 16
    num_attention_heads: int = 8

    # -- Point (XYZ) Embedder
    embed_frequency: int = 8
    embed_include_pi: bool = False

    # -- Framestep/Alpha Embedder
    framestep_encoding_strategy: str = "linear"
    num_freqs_ts: int = 128

    # -- Output
    prediction_mode: str = "direct"  # direct | residual

    def __post_init__(self):
        super().__init__()
        self.has_extra_query_feats = self.in_extra_channels > 0
        self.width_per_head = self.width // self.num_attention_heads

        # -- Framestep + Alpha Embedder
        self.timestep_embedder = TimestepEmbedder(
            frequency_embedding_size=self.width // 2,
        )

        # -- Point (XYZ) Embedder
        self.embedder = FrequencyPositionalEmbedding(
            num_freqs=self.embed_frequency,
            logspace=True,
            input_dim=self.in_channels,
            include_pi=self.embed_include_pi,
        )

        # -- Decoder
        self.input_dim_decoder = self.embedder.out_dim + self.in_extra_channels

        self.blocks = nn.ModuleList(
            [
                # -- N Self Attention layers
                FlowMatchingBlock(
                    dim=self.width,
                    num_attention_heads=self.num_attention_heads,
                    use_self_attention=True,
                    use_cross_attention=False,
                    attention_qk_norm=None,
                    attention_bias=False,
                    ff_activation="gelu",
                )
                for _ in range(self.num_layers)
            ]
            + [
                # -- 1 Cross Attention layer
                FlowMatchingBlock(
                    dim=self.width,
                    num_attention_heads=self.num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=self.width,
                    cross_attention_norm_type="layer_norm",
                    attention_qk_norm=None,
                    attention_bias=False,
                    ff_activation="gelu",
                )
            ]
        )
        self.proj_query = nn.Linear(self.input_dim_decoder, self.width, bias=True)
        self.norm_out = nn.LayerNorm(self.width)
        self.proj_out = nn.Linear(self.width, self.out_dim, bias=True)
        self.post_quant = nn.Linear(self.latent_channels, self.width, bias=True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def apply_displacement(
        self,
        vertex: torch.Tensor,
        displacement: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply predicted displacement field to source vertices.

        Args:
            vertex (B, V, 3): Source mesh vertices (reference frame).
            displacement (B, T_out, V, 3): Predicted displacement field.

        Returns:
            vertex_deformed (B, T_out, V, 3): Deformed vertices clamped to [-1, 1].
        """
        if self.prediction_mode == "direct":
            return torch.clamp(displacement, min=-1.0, max=1.0)
        elif self.prediction_mode == "residual":
            return torch.clamp(vertex[:, None] + displacement, min=-1.0, max=1.0)
        else:
            raise ValueError(f"Invalid prediction_mode: {self.prediction_mode}")

    def fwd_kv_cache(
        self,
        hidden_states: torch.Tensor,
        freqs_rot: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        for _, block in enumerate(self.blocks[:-1]):
            hidden_states = block(hidden_states, freqs_rot=freqs_rot)
        return hidden_states

    def fwd_cross_attn(
        self,
        kv_cache: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.proj_query(queries)
        logits = self.blocks[-1](queries, encoder_hidden_states=kv_cache)
        logits = self.proj_out(self.norm_out(logits))
        logits = logits * -1
        return logits

    def forward(
        self,
        latent: torch.Tensor,
        framestep: torch.Tensor,
        source_alpha: torch.Tensor,
        target_alphas: torch.Tensor,
        query: torch.Tensor,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """
        Given a set of latents (from stage I) and query points (mesh vertices at normalized time source_alpha),
        ActionMesh autoencoder (stage II) predicts how each vertex should move from source_alpha to each target_alpha timestep.

        Shape legend:
            B: batch size
            T: number of frames
            N: number of tokens per frame
            D: latent dimension
            T_out: number of target timesteps to predict
            V: number of query vertices

        Args:
            latent (B, T, N, D): Denoised latents from Stage I denoiser.
            framestep (B, T): Video timesteps for temporal positional encoding.
            source_alpha (B): Source timestep in normalized time [0, 1].
            target_alphas (B, T_out): Target timesteps in normalized time [0, 1].
            query (B, V, 3|6): Query points (vertices, optionally with normals).
            step_callback: Optional callback called at each step with (step, total_steps).

        Returns:
            displacement (B, T_out, V, 3): Per-vertex displacement field in [-1, 1] from source_alpha to each target_alpha timestep.
        """
        assert target_alphas.ndim == 2
        assert source_alpha.ndim == 1
        B, T, N, _ = latent.shape
        _, T_out = target_alphas.shape
        V = query.shape[1]

        framestep_centered = merge_batch_time(
            scale_timestep(framestep, center=True, scale=False)
        )

        # -- Project latents to model width
        latent_proj = merge_time_tokens(self.post_quant(latent))

        # -- Compute rotary positional encoding from framesteps
        freqs_cos, freqs_sin = compute_rotary_embeddings(
            embed_dim=self.width_per_head,
            positions=framestep_centered,
        )
        freqs_cos = freqs_cos.reshape((B, T, -1)).to(latent)  # (B, T, D)
        freqs_sin = freqs_sin.reshape((B, T, -1)).to(latent)

        # Expand freqs for latent tokens (N per frame) + alpha token (1 per frame)
        # Using repeat_interleave: each frame's freq is repeated N times for N tokens
        freqs_cos = torch.cat(
            [
                freqs_cos.repeat_interleave(N, dim=1),  # (B, T*N, D) - 3D latents
                freqs_cos,  # (B, T, D) - alpha latents
            ],
            dim=1,
        )  # (B, T*N + T, D)
        freqs_sin = torch.cat(
            [
                freqs_sin.repeat_interleave(N, dim=1),  # (B, T*N, D) - 3D latents
                freqs_sin,  # (B, T, D) - alpha latents
            ],
            dim=1,
        )  # (B, T*N + T, D)

        # -- Build alpha conditioning pairs
        source_alphas = source_alpha.unsqueeze(-1).expand_as(target_alphas)
        alpha_embedded = self.timestep_embedder(source_alphas, target_alphas)[
            :, None
        ].repeat(1, T, 1, 1)

        # -- Embed query points + (optionally have normals)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            query_embed = self.embedder(query[..., :3])
            if self.has_extra_query_feats:
                query_embed = torch.cat([query_embed, query[..., 3:]], dim=-1)

        # -- Predict displacement for each target timestep
        displacements = torch.empty(
            (B, T_out, V, self.out_dim), dtype=latent.dtype, device=latent.device
        )
        for i in tqdm(
            range(T_out),
            disable=not self.verbose,
            desc="Temporal 3D Decoding (Stage II)",
        ):
            if step_callback is not None:
                step_callback(i + 1, T_out)

            # Prepend alpha embedding to latent sequence
            latent_with_alpha = torch.cat([latent_proj, alpha_embedded[:, :, i]], dim=1)

            # Self-attention blocks with rotary temporal encoding
            kv_cache = self.fwd_kv_cache(
                latent_with_alpha, freqs_rot=(freqs_cos, freqs_sin)
            )

            # Cross-attention with query points
            with torch.amp.autocast(device_type="cuda", enabled=False):
                displacements[:, i] = self.fwd_cross_attn(kv_cache, query_embed)

        return 2 * torch.sigmoid(displacements) - 1.0
