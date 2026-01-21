# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def compute_rotary_embeddings(
    embed_dim: int,
    positions: torch.Tensor,
    base_freq: float = 10000.0,
    freq_scale: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rotary positional embeddings (RoPE) for transformer attention.

    Generates sinusoidal frequency tensors used to inject positional information
    into attention queries and keys via rotation in the complex plane.

    Shape legend:
        S: sequence length (number of positions)
        D: embedding dimension (must be even)

    Args:
        embed_dim (int): Dimension of the rotary embedding. Must be even.
        positions (S,): Position values for each element in the sequence (e.g., video timesteps).
        base_freq: Base frequency for the sinusoidal encoding.
        freq_scale: Scaling factor for frequency computation (for context length extrapolation).
        dtype: Data type for intermediate frequency computation (float32 or float64).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (cos_embed, sin_embed)
            - cos_embed (S, D): Cosine component of rotary embeddings.
            - sin_embed (S, D): Sine component of rotary embeddings.
    """
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"

    # Compute inverse frequencies: [D/2]
    inv_freq = (
        1.0
        / (
            base_freq
            ** (
                torch.arange(0, embed_dim, 2, dtype=dtype, device=positions.device)
                / embed_dim
            )
        )
        / freq_scale
    )

    # Compute phase values: θ = position × frequency -> [S, D/2]
    phases = torch.outer(positions, inv_freq)

    # Expand to full dimension by repeating each frequency: [S, D]
    cos_embed = (
        phases.cos()
        .repeat_interleave(2, dim=1, output_size=phases.shape[1] * 2)
        .float()
    )
    sin_embed = (
        phases.sin()
        .repeat_interleave(2, dim=1, output_size=phases.shape[1] * 2)
        .float()
    )

    return cos_embed, sin_embed


def apply_rotary_embedding(
    x: torch.Tensor,
    cos_embed: torch.Tensor,
    sin_embed: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings (RoPE) to query or key tensors.

    Rotates pairs of adjacent dimensions in the input tensor using precomputed
    cos/sin embeddings, injecting positional information into attention.

    Shape legend:
        B: batch size
        H: number of attention heads
        S: sequence length
        D: head dimension (must be even)

    Args:
        x (B, H, S, D): Query or key tensor to apply rotary embeddings to.
        cos_embed (S, D) or (B, S, D): Cosine component of rotary embeddings.
        sin_embed (S, D) or (B, S, D): Sine component of rotary embeddings.

    Returns:
        torch.Tensor (B, H, S, D): Input tensor with rotary embeddings applied.
    """
    cos, sin = cos_embed, sin_embed
    assert cos.ndim == sin.ndim

    # Expand dimensions for broadcasting with [B, H, S, D]
    if cos.ndim == 2:
        # [S, D] -> [1, 1, S, D]
        cos = cos[None, None]
        sin = sin[None, None]
    elif cos.ndim == 3:
        # [B, S, D] -> [B, 1, S, D]
        cos = cos[:, None]
        sin = sin[:, None]
    else:
        raise ValueError(f"cos_embed/sin_embed should be 2D or 3D, got {cos.ndim}D.")

    cos, sin = cos.to(x.device), sin.to(x.device)

    # Split into real/imaginary pairs and rotate
    # [B, H, S, D] -> [B, H, S, D//2, 2] -> unbind -> [B, H, S, D//2] each
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)

    # Rotate: [-x_imag, x_real] represents 90° rotation in complex plane
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    # Apply rotation: x * cos + rotate(x) * sin
    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    return out
