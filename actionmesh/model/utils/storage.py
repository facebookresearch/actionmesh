# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import torch
import trimesh

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class TimestepIndexedStorage(ABC, Generic[T]):
    """
    Base class for storing items indexed by timestep.

    Provides common functionality for adding, retrieving, and ordering items
    by their associated timesteps.
    """

    items: list[T] = field(default_factory=list)
    timesteps: list[float] = field(default_factory=list)
    verbose: bool = False
    tag: str = ""

    @property
    def n_timesteps(self) -> int:
        """Number of stored timesteps."""
        return len(self.timesteps)

    def _log_prefix(self) -> str:
        """Log prefix for verbose output."""
        name = self.__class__.__name__.upper()
        return f"{name} {self.tag}" if self.tag else name

    @abstractmethod
    def _get_empty_item(self, device: str) -> T:
        """Return an empty/default item for missing items."""
        pass

    def get_timestep_index(self, timestep: float, eps: float = 1e-5) -> int | None:
        """Find index of a timestep, or None if not found."""
        for index, ts in enumerate(self.timesteps):
            if abs(ts - timestep) < eps:
                return index
        return None

    def _update_one(
        self, timestep: float, item: T, replace: bool = False
    ) -> tuple[bool, bool]:
        """Update a single item. Returns (added, replaced) flags."""
        index = self.get_timestep_index(timestep)
        if index is None:
            self.timesteps.append(timestep)
            self.items.append(item)
            return True, False
        elif replace:
            self.items[index] = item
            return False, True
        return False, False

    def _log_updates(self, added: list[float], replaced: list[float]) -> None:
        """Log added/replaced timesteps if verbose mode is enabled."""
        if self.verbose:
            if added:
                logger.info(f"[{self._log_prefix()}] Added timesteps {added}")
            if replaced:
                logger.info(f"[{self._log_prefix()}] Replaced timesteps {replaced}")

    def get_ordered_timesteps(self) -> torch.Tensor:
        """Return all timesteps as a sorted tensor."""
        ordered_indices = sorted(
            range(len(self.timesteps)), key=lambda i: self.timesteps[i]
        )
        return torch.tensor([self.timesteps[i] for i in ordered_indices])

    def _get_ordered_indices(self) -> list[int]:
        """Return all indices that would sort timesteps in ascending order."""
        return sorted(range(len(self.timesteps)), key=lambda i: self.timesteps[i])


@dataclass
class LatentBank(TimestepIndexedStorage[torch.Tensor]):
    """Storage for latent tensors indexed by timestep."""

    empty_dims: tuple[int, int] = field(default_factory=lambda: (768, 64))

    def __post_init__(self):
        self.n_latent_dims = len(self.empty_dims)

    def _get_empty_item(self, device: str) -> torch.Tensor:
        """Return a zero tensor for missing latents."""
        return torch.zeros(self.empty_dims, dtype=torch.float32, device=device)

    def update(
        self, timesteps: torch.Tensor, latents: torch.Tensor, replace: bool = False
    ) -> None:
        """
        Update the bank with new latents.

        Args:
            timesteps (N,): Timesteps tensor.
            latents (*, *empty_dims): Latent tensor with any leading dims, as long as
                the total number of elements matches len(timesteps).
            replace: Whether to replace existing items at matching timesteps.
        """
        timesteps = timesteps.flatten()
        n = timesteps.shape[0]

        # Reshape latents to (N, *empty_dims) regardless of input shape
        latents = latents.reshape(n, *self.empty_dims)

        added, replaced = [], []
        for i in range(timesteps.shape[0]):
            was_added, was_replaced = self._update_one(
                timestep=timesteps[i].item(),
                item=latents[i],
                replace=replace,
            )
            if was_added:
                added.append(timesteps[i].item())
            if was_replaced:
                replaced.append(timesteps[i].item())

        self._log_updates(added, replaced)

    def get(
        self, timesteps: torch.Tensor, device: str, add_batch_dim: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve latents for given timesteps.

        Args:
            timesteps (N,): Timesteps to retrieve.
            device: Device to place output tensors on.
            add_batch_dim: If True, add a leading batch dimension to outputs.

        Returns:
            latents: Retrieved latents. Shape (N, *empty_dims) or (1, N, *empty_dims).
            masks: Binary mask, 1 where latent exists, 0 otherwise.
                Shape (N,) or (1, N).
        """
        assert timesteps.ndim == 1

        latents, masks = [], []
        for timestep in timesteps:
            index = self.get_timestep_index(timestep)
            if index is None:
                latents.append(self._get_empty_item(device))
                masks.append(0)
            else:
                latents.append(self.items[index].to(device))
                masks.append(1)

        latents_out = torch.stack(latents)
        masks_out = torch.tensor(masks, dtype=torch.int32, device=device)

        if add_batch_dim:
            return latents_out[None], masks_out[None]
        return latents_out, masks_out

    def get_ordered(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve all latents ordered by timestep.

        Returns:
            latents (N, *empty_dims): All stored latents sorted by timestep.
            timesteps (N,): Corresponding timesteps in ascending order.
        """
        ordered_indices = self._get_ordered_indices()
        ordered_latents = torch.stack([self.items[i] for i in ordered_indices])
        ordered_timesteps = torch.tensor(
            [self.timesteps[i] for i in ordered_indices]
        ).to(ordered_latents)
        return ordered_latents, ordered_timesteps


@dataclass
class MeshBank(TimestepIndexedStorage[trimesh.Trimesh]):
    """Storage for meshes indexed by timestep."""

    def _get_empty_item(self, device: str) -> trimesh.Trimesh | None:
        """Return None for missing meshes."""
        return None

    def update(
        self,
        timesteps: torch.Tensor,
        meshes: list[trimesh.Trimesh],
        replace: bool = False,
    ) -> None:
        """
        Update the bank with new meshes.

        Args:
            timesteps (N,): Timesteps tensor.
            meshes (N,): List of meshes.
            replace: Whether to replace existing items at matching timesteps.
        """
        assert timesteps.shape[0] == len(meshes)
        assert timesteps.ndim == 1

        added, replaced = [], []
        for i in range(timesteps.shape[0]):
            was_added, was_replaced = self._update_one(
                timestep=timesteps[i].item(),
                item=meshes[i],
                replace=replace,
            )
            if was_added:
                added.append(timesteps[i].item())
            if was_replaced:
                replaced.append(timesteps[i].item())

        self._log_updates(added, replaced)

    def get(self, timesteps: torch.Tensor) -> list[trimesh.Trimesh | None]:
        """
        Retrieve meshes for given timesteps.

        Args:
            timesteps (N,): Timesteps to retrieve.

        Returns:
            meshes (N,): List of retrieved meshes (None for missing timesteps).
        """
        assert timesteps.ndim == 1
        return [
            (
                self.items[index]
                if (index := self.get_timestep_index(ts)) is not None
                else None
            )
            for ts in timesteps
        ]

    def get_ordered(self, device: str) -> tuple[list[trimesh.Trimesh], torch.Tensor]:
        """
        Retrieve all meshes ordered by timestep.

        Args:
            device: Device to place timesteps tensor on.

        Returns:
            meshes (N,): All stored meshes sorted by timestep.
            timesteps (N,): Corresponding timesteps in ascending order.
        """
        ordered_indices = self._get_ordered_indices()
        ordered_meshes = [self.items[i] for i in ordered_indices]
        ordered_timesteps = torch.tensor(
            [self.timesteps[i] for i in ordered_indices],
            device=device,
        )
        return ordered_meshes, ordered_timesteps
