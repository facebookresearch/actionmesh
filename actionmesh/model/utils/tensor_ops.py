# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensor reshaping utilities for batched temporal sequences.

Shape convention: (B, T, N, D)
    B: Batch size
    T: Number of timesteps/frames
    N: Number of tokens
    D: Feature dimension
"""

import torch
from einops import rearrange

# =============================================================================
# Flatten/Unflatten batch and time: (B, T, N, D) <-> (B*T, N, D)
# =============================================================================


def merge_batch_time(x: torch.Tensor) -> torch.Tensor:
    """
    Merge batch and time dimensions.

    Args:
        x (B, T, N, D): Input tensor.

    Returns:
        (B*T, N, D): Merged tensor.
    """
    return rearrange(x, "b t ... -> (b t) ...")


def split_batch_time(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """
    Split merged batch-time dimension.

    Args:
        x (B*T, N, D): Merged tensor.
        n_frames: Number of frames T.

    Returns:
        (B, T, N, D): Split tensor.
    """
    return rearrange(x, "(b t) ... -> b t ...", t=n_frames)


# =============================================================================
# Flatten/Unflatten time and tokens: (B, T, N, D) <-> (B, T*N, D)
# =============================================================================


def merge_time_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    Merge time and token dimensions.

    Args:
        x (B, T, N, D): Input tensor.

    Returns:
        (B, T*N, D): Merged tensor.
    """
    return rearrange(x, "b t n ... -> b (t n) ...")


def split_time_tokens(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """
    Split merged time-token dimension.

    Args:
        x (B, T*N, D): Merged tensor.
        n_frames: Number of frames T.

    Returns:
        (B, T, N, D): Split tensor.
    """
    return rearrange(x, "b (t n) ... -> b t n ...", t=n_frames)


# =============================================================================
# Cross-frame attention reshaping: (B*T, N, D) <-> (B, T*N, D)
# =============================================================================


def flat_batch_to_flat_seq(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """
    Convert flat-batch to flat-sequence for cross-frame attention.

    Args:
        x (B*T, N, D): Flat batch tensor.
        n_frames: Number of frames T.

    Returns:
        (B, T*N, D): Flat sequence tensor.
    """
    return rearrange(x, "(b t) n ... -> b (t n) ...", t=n_frames)


def flat_seq_to_flat_batch(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """
    Convert flat-sequence back to flat-batch.

    Args:
        x (B, T*N, D): Flat sequence tensor.
        n_frames: Number of frames T.

    Returns:
        (B*T, N, D): Flat batch tensor.
    """
    return rearrange(x, "b (t n) ... -> (b t) n ...", t=n_frames)
