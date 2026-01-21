# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from actionmesh.model.utils.rotary_embedding import apply_rotary_embedding
from actionmesh.model.utils.tensor_ops import (
    flat_batch_to_flat_seq,
    flat_seq_to_flat_batch,
)
from diffusers.models.attention_processor import Attention


class AttentionProcessor:
    r"""
    Processor for implementing the scaled dot-product attention.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        inflate_self_attention: bool = False,
        freqs_rot: Optional[torch.Tensor] = None,
        n_frames: Optional[int] = None,
    ) -> torch.Tensor:

        # -- Inflate self-attention if needed
        if inflate_self_attention:
            assert n_frames is not None
            hidden_states = flat_batch_to_flat_seq(
                hidden_states,
                n_frames=n_frames,
            )
            if freqs_rot is not None:
                freqs_rot = (
                    flat_batch_to_flat_seq(
                        freqs_rot[0],
                        n_frames=n_frames,
                    ),
                    flat_batch_to_flat_seq(
                        freqs_rot[1],
                        n_frames=n_frames,
                    ),
                )

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        # -- Compute Query
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        # -- Compute Key and Value
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # -- Split heads
        if not attn.is_cross_attention:
            qkv = torch.cat((query, key, value), dim=-1)
            split_size = qkv.shape[-1] // attn.heads // 3
            qkv = qkv.view(batch_size, -1, attn.heads, split_size * 3)
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            kv = torch.cat((key, value), dim=-1)
            split_size = kv.shape[-1] // attn.heads // 2
            kv = kv.view(batch_size, -1, attn.heads, split_size * 2)
            key, value = torch.split(kv, split_size, dim=-1)
        head_dim = key.shape[-1]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # -- Apply RoPE if needed
        if freqs_rot is not None:
            cos_embed, sin_embed = freqs_rot
            query = apply_rotary_embedding(query, cos_embed, sin_embed)
            key = apply_rotary_embedding(key, cos_embed, sin_embed)

        # -- Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # -- Linear + Dropout + Residual
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # -- De-Inflate
        if inflate_self_attention:
            assert n_frames is not None
            hidden_states = flat_seq_to_flat_batch(
                hidden_states,
                n_frames=n_frames,
            )

        return hidden_states
