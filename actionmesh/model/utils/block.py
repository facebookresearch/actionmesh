# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

import torch
from actionmesh.model.utils.attention_processor import AttentionProcessor
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
from torch import nn


@dataclass(eq=False)
class FlowMatchingBlock(nn.Module):
    """
    Transformer block for Flow Matching models with optional self-attention,
    cross-attention, feed-forward layers, and U-Net style skip connections.

    Args:
        dim: Hidden dimension size.
        num_attention_heads: Number of attention heads.
        use_self_attention: Enable self-attention layer.
        inflate_self_attention: Apply inflated attention for temporal modeling.
        use_cross_attention: Enable cross-attention layer.
        cross_attention_dim: Dimension of cross-attention context (required if use_cross_attention=True).
        cross_attention_norm_type: Normalization type for cross-attention.
        attention_qk_norm: Normalization type for query/key projections.
        attention_bias: Use bias in QKV projections.
        ff_inner_dim: Feed-forward hidden dimension (defaults to 4x dim).
        ff_bias: Use bias in feed-forward layers.
        ff_activation: Activation function for feed-forward.
        skip: Enable U-Net style skip connection input.
    """

    dim: int

    # -- Self/Cross-Attention
    num_attention_heads: int
    use_self_attention: bool = True
    inflate_self_attention: bool = False
    use_cross_attention: bool = True
    cross_attention_dim: Optional[int] = None
    cross_attention_norm_type: str = "fp32_layer_norm"
    attention_qk_norm: Optional[str] = "rms_norm"
    attention_bias: bool = True

    # -- Linear
    ff_inner_dim: Optional[int] = None
    ff_bias: bool = True
    ff_activation: str = "gelu"

    # -- Skip
    skip: bool = False

    def __post_init__(self):
        super().__init__()

        if self.use_self_attention:
            self.norm_s_attn = FP32LayerNorm(
                self.dim, eps=1e-5, elementwise_affine=True
            )
            self.s_attn = Attention(
                query_dim=self.dim,
                cross_attention_dim=None,
                dim_head=self.dim // self.num_attention_heads,
                heads=self.num_attention_heads,
                qk_norm=self.attention_qk_norm,
                eps=1e-6,
                bias=self.attention_bias,
                residual_connection=False,
                processor=AttentionProcessor(),
            )

        if self.use_cross_attention:
            assert (
                self.cross_attention_dim is not None
            ), "cross_attention_dim is required when use_cross_attention=True"
            self.norm_x_attn = FP32LayerNorm(
                self.dim, eps=1e-5, elementwise_affine=True
            )
            self.x_attn = Attention(
                query_dim=self.dim,
                cross_attention_dim=self.cross_attention_dim,
                dim_head=self.dim // self.num_attention_heads,
                heads=self.num_attention_heads,
                qk_norm=self.attention_qk_norm,
                cross_attention_norm=self.cross_attention_norm_type,
                eps=1e-6,
                bias=self.attention_bias,
                processor=AttentionProcessor(),
            )

        self.norm_ff = FP32LayerNorm(self.dim, eps=1e-5, elementwise_affine=True)
        self.ff = FeedForward(
            self.dim,
            activation_fn=self.ff_activation,
            inner_dim=self.ff_inner_dim,
            bias=self.ff_bias,
        )

        if self.skip:
            self.norm_skip = FP32LayerNorm(self.dim, eps=1e-5, elementwise_affine=True)
            self.linear_skip = nn.Linear(2 * self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        n_frames: Optional[int] = None,
        freqs_rot: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (B, N, D): Input tensor.
            encoder_hidden_states (B, S, cross_attention_dim): Context for cross-attention.
            n_frames: Number of frames for inflated temporal attention.
            freqs_rot [(B, N, rot_embed_dim), (B, N, rot_embed_dim)]: Rotary position embeddings as (cos, sin) tuple.
            skip (B, N, D): Skip connection input.

        Returns:
            hidden_states (B, N, D).
        """

        # -- Skip connection + skip linear
        if self.skip:
            cat = torch.cat([skip, hidden_states], dim=-1)
            hidden_states = self.norm_skip(self.linear_skip(cat))

        # -- Self-attention
        if self.use_self_attention:
            hidden_states = hidden_states + self.s_attn(
                self.norm_s_attn(hidden_states),
                n_frames=n_frames,
                inflate_self_attention=self.inflate_self_attention,
                freqs_rot=freqs_rot,
            )

        # -- Cross-attention
        if self.use_cross_attention:
            hidden_states = hidden_states + self.x_attn(
                self.norm_x_attn(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
            )

        # -- Feed-forward
        hidden_states = hidden_states + self.ff(self.norm_ff(hidden_states))

        return hidden_states
