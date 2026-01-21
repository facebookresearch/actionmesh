# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(eq=False)
class FrequencyPositionalEmbedding(nn.Module):
    """Sin/cosine positional embedding using frequency encoding."""

    input_dim: int = 3
    num_freqs: int = 6
    logspace: bool = True
    include_input: bool = True
    include_pi: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        frequencies = self._build_frequencies()
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def out_dim(self):
        extra = 1 if self.include_input or self.num_freqs == 0 else 0
        return self.input_dim * (self.num_freqs * 2 + extra)

    def _build_frequencies(self) -> torch.Tensor:
        if self.logspace:
            freqs = 2.0 ** torch.arange(self.num_freqs, dtype=torch.float32)
        else:
            freqs = torch.linspace(
                1.0, 2.0 ** (self.num_freqs - 1), self.num_freqs, dtype=torch.float32
            )
        if self.include_pi:
            freqs *= torch.pi
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs == 0:
            return x

        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal encoding.

    Supports arbitrary input dimensions:
    - Input shape: (*) -> output shape: (*, output_dim)

    Note:
        frequency_embedding_size must be even.
    """

    def __init__(
        self,
        frequency_embedding_size: int = 256,
        max_period: int = 10_000,
    ):
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError(
                f"frequency_embedding_size must be even, got {frequency_embedding_size}"
            )

        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        # Precompute and cache frequencies as a buffer (not a parameter)
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, *timesteps: torch.Tensor) -> torch.Tensor:
        """
        Embed one or more timestep tensors and concatenate their embeddings.

        All input tensors must have the same shape (*). Each is embedded separately
        and the results are concatenated along the last dimension.

        Args:
            *timesteps: One or more tensors of shape (*) containing timestep values.
                        All tensors must have the same shape.

        Returns:
            Tensor of shape (*, N * output_dim) where N is the number of input tensors.

        Examples:
            >>> embedder = TimestepEmbedder(frequency_embedding_size=256)
            >>> t1 = torch.tensor([1.0, 2.0])  # shape: (2,)
            >>> out = embedder(t1)              # shape: (2, 256)
            >>> t2 = torch.tensor([3.0, 4.0])  # shape: (2,)
            >>> out = embedder(t1, t2)          # shape: (2, 512)
        """
        if len(timesteps) == 0:
            raise ValueError("At least one timestep tensor must be provided")

        n_timesteps = len(timesteps)
        t0 = timesteps[0]
        half = self.frequency_embedding_size // 2
        dim = self.frequency_embedding_size

        # Pre-allocate output tensor for all embeddings
        out_shape = (*t0.shape, n_timesteps * dim)
        output = torch.empty(out_shape, dtype=torch.float32, device=t0.device)

        # Embed each timestep directly into the output tensor
        for i, t in enumerate(timesteps):
            args = t[..., None].float() * self.freqs
            output[..., i * dim : i * dim + half] = torch.cos(args)
            output[..., i * dim + half : (i + 1) * dim] = torch.sin(args)

        return output

    @property
    def output_dim(self):
        return self.frequency_embedding_size


def scale_timestep(
    timestep: torch.Tensor, center: bool = True, scale: bool = False
) -> torch.Tensor:
    """
    Center and Scale timestep
    Args:
        - timestep: (B, N_TS)
    Returns:
        - timesteps_centered_scaled: (B, N_TS)
    """
    timestep_min = timestep.min(dim=1).values
    timestep_max = timestep.max(dim=1).values

    if center:
        timestep = timestep - timestep_min.unsqueeze(1)
    if scale:
        timestep = timestep / (timestep_max - timestep_min).unsqueeze(1)

    return timestep


def get_n_subdivisions(start: int, end: int, level: int = 1) -> int:
    """
    Compute number of points after recursive subdivision.

    Args:
        start: Start index.
        end: End index.
        level: Number of subdivision levels.

    Returns:
        Total number of points.
    """
    n_points = int(end - start + 1)
    for _ in range(1, level):
        n_points += n_points - 1
    return n_points


def interpolate_timesteps(
    timesteps: torch.Tensor,
    subsampling_level: int,
    device: torch.device | str,
    drop_first: bool = False,
) -> torch.Tensor:
    """
    Generate interpolated timesteps between min and max of input timesteps.

    Args:
        timesteps (*, T): Input timesteps (any shape, uses global min/max).
        subsampling_level: Subsampling level for computing number of steps.
        device: Device to place output tensor on.

    Returns:
        interpolated (1, n_steps): Linearly interpolated timesteps from min to max.
    """
    t_min = timesteps.min().item()
    t_max = timesteps.max().item()
    n_steps = get_n_subdivisions(t_min, t_max, level=subsampling_level)

    output = torch.linspace(t_min, t_max, n_steps, device=device).reshape(1, -1)

    if drop_first:
        output = output[:, 1:]
    return output
