# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import trimesh

logger = logging.getLogger(__name__)


@contextmanager
def scoped_seed(seed: int | None):
    """Context manager that temporarily sets numpy and python random seeds."""
    if seed is None:
        yield
        return
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


def get_mesh_features(mesh: trimesh.Trimesh, with_normals: bool) -> torch.Tensor:
    """
    Extract vertex positions and optionally normals from a mesh.

    Args:
        mesh: Input triangle mesh.
        with_normals: If True, concatenate normalized vertex normals.

    Returns:
        features (V, 3|6): Vertex positions, optionally with normals.
    """
    features = torch.from_numpy(mesh.vertices).float()
    if with_normals:
        normals = torch.from_numpy(mesh.vertex_normals.copy()).float()
        normals = torch.nn.functional.normalize(normals, p=2, dim=-1)
        features = torch.cat([features, normals], dim=-1)
    return features


def remove_degenerate_faces(mesh: trimesh.Trimesh) -> None:
    """Remove degenerate faces from a mesh (compatible with old and new trimesh versions)."""
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())


def remove_duplicate_faces(mesh: trimesh.Trimesh) -> None:
    """Remove duplicate faces from a mesh (compatible with old and new trimesh versions)."""
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    else:
        mesh.update_faces(mesh.unique_faces())


def remove_unreferenced_vertices(mesh: trimesh.Trimesh) -> None:
    """Remove unreferenced vertices from a mesh (compatible with old and new trimesh versions)."""
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    else:
        mesh.update_vertices(mask=mesh.referenced_vertices)


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int = 40_000,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """Decimate a mesh using quadric decimation to reduce face count.

    Args:
        mesh: Trimesh object to decimate.
        target_faces: Target number of faces. Mesh with fewer faces is unchanged.
        verbose: If True, log before/after statistics.

    Returns:
        Decimated mesh.
    """
    original_faces = len(mesh.faces)

    if original_faces <= target_faces:
        if verbose:
            logger.info(
                f"[Decimation] Skipped: mesh has {original_faces:,} faces "
                f"(<= target {target_faces:,})"
            )
        return mesh

    if verbose:
        logger.info(f"[Decimation] Before: {original_faces:,} faces")

    decimated = mesh.simplify_quadric_decimation(face_count=target_faces)

    if verbose:
        logger.info(f"[Decimation] After: {len(decimated.faces):,} faces")

    return decimated


def remove_floaters(
    mesh: trimesh.Trimesh,
    threshold: float = 0.0,
) -> trimesh.Trimesh:
    """Remove small disconnected components (floaters) from a mesh.

    Args:
        mesh: Trimesh object to clean.
        threshold: Minimum size as fraction of largest component. Components with
            fewer faces than (largest_component_faces * threshold) are removed.

    Returns:
        Cleaned mesh.
    """
    components = mesh.split(only_watertight=False)
    num_components = len(components)

    if num_components <= 1:
        logger.debug(f"[Floaters] Skipped: mesh has {num_components} component(s)")
        return mesh

    max_faces = max(len(c.faces) for c in components)
    min_faces = int(max_faces * threshold)
    kept = [c for c in components if len(c.faces) >= min_faces]

    if not kept:
        logger.warning(
            f"[Floaters] No components kept after filtering "
            f"(threshold={threshold}, min_faces={min_faces}), returning original mesh"
        )
        return mesh

    logger.info(
        f"[Floaters] Removed {num_components - len(kept)} component(s): "
        f"{num_components} -> {len(kept)}"
    )

    return trimesh.util.concatenate(kept)


def normalize_mesh_to_bounds(
    mesh: trimesh.Trimesh,
    bounds: tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    ),
) -> trimesh.Trimesh:
    """Rescale mesh only if its bounding box exceeds the specified bounds.

    Args:
        mesh: Trimesh object to normalize.
        bounds: Target bounds as (min_x, min_y, min_z, max_x, max_y, max_z).

    Returns:
        Original mesh if within bounds, otherwise a new mesh scaled to fit.
    """
    target_min = np.array(bounds[:3])
    target_max = np.array(bounds[3:])
    target_size = target_max - target_min

    mesh_min, mesh_max = mesh.bounds
    mesh_size = mesh_max - mesh_min

    # Check if mesh is already within bounds
    if np.all(mesh_min >= target_min) and np.all(mesh_max <= target_max):
        return mesh

    # Scale uniformly to fit within bounds (only if exceeding)
    scale = min(1.0, (target_size / np.maximum(mesh_size, 1e-8)).min())

    target_center = (target_min + target_max) / 2
    mesh_center = (mesh_min + mesh_max) / 2

    new_vertices = (mesh.vertices - mesh_center) * scale + target_center

    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False,
    )


@dataclass(eq=False)
class MeshPostprocessor:
    bounds: tuple[float, float, float, float, float, float] = (
        -1.005,
        -1.005,
        -1.005,
        1.005,
        1.005,
        1.005,
    )
    face_decimation: int = -1
    floaters_threshold: float = 0.0
    verbose: bool = True

    def __post_init__(self):
        assert self.bounds[0] == self.bounds[1] == self.bounds[2]
        assert self.bounds[3] == self.bounds[4] == self.bounds[5]

    @torch.no_grad()
    def process_mesh(
        self,
        mesh: trimesh.Trimesh,
        seed: int | None = None,
    ) -> trimesh.Trimesh:
        """
        Post-process a single Trimesh mesh (decimation, floaters removal, etc.)

        Args:
            mesh: Input trimesh to process.
            seed: Random seed for reproducibility.

        Returns:
            Processed trimesh.
        """
        with scoped_seed(seed):
            # Clean up mesh topology
            mesh.merge_vertices()  # Merge duplicates
            remove_degenerate_faces(mesh)  # Remove bad faces
            remove_duplicate_faces(mesh)  # Remove identical faces
            remove_unreferenced_vertices(mesh)  # Clean up unused vertices

            # -- Mesh decimation
            if self.face_decimation != -1:
                mesh = decimate_mesh(
                    mesh, target_faces=self.face_decimation, verbose=self.verbose
                )

            # -- Remove floaters
            if self.floaters_threshold > 0.0:
                mesh = remove_floaters(mesh, threshold=self.floaters_threshold)

        return mesh
