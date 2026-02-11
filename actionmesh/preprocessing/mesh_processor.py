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
from scipy.spatial import cKDTree

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


def merge_and_clean_mesh(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate vertices and clean up mesh topology in-place.

    GLB loading creates duplicate vertices at UV seams and hard edges.
    This function merges them and removes degenerate/duplicate faces
    and unreferenced vertices.

    It also returns a mapping from the original (pre-merge) vertex indices
    to the merged vertex indices, so callers can expand the merged
    vertices back to the original topology when texture / UV must be
    preserved.

    Args:
        mesh: Trimesh object to clean (modified in-place).

    Returns:
        vertex_merge_map: int array of shape (N_original,) where
            ``vertex_merge_map[i]`` is the index of the merged vertex
            that original vertex *i* maps to.
        pre_merge_faces: int array of shape (F_original, 3), the face
            array before merging (uses original vertex indices).
    """

    pre_merge_verts = mesh.vertices.copy()
    pre_merge_faces = mesh.faces.copy()

    mesh.merge_vertices()
    remove_degenerate_faces(mesh)
    remove_duplicate_faces(mesh)
    remove_unreferenced_vertices(mesh)

    # For each original vertex, find its index in the merged mesh via
    # nearest-neighbour lookup.  This is robust to floating-point
    # tolerance differences between our code and trimesh's internal
    # merge logic.
    tree = cKDTree(mesh.vertices)
    distances, vertex_merge_map = tree.query(pre_merge_verts)
    assert np.all(distances < 1e-6), (
        "Some pre-merge vertices have no close match in the "
        f"merged mesh (max dist={distances.max():.2e}). "
        "merge_vertices() may have altered positions."
    )

    return vertex_merge_map, pre_merge_faces


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


# ---------------------------------------------------------------------------
# Mesh normalization
# ---------------------------------------------------------------------------


@dataclass
class NormalizationParams:
    """Parameters that describe the normalization applied to a mesh."""

    bbox_center: np.ndarray | None
    scale: float


def normalize_mesh(
    mesh: trimesh.Trimesh,
    center: bool = True,
) -> tuple[trimesh.Trimesh, NormalizationParams]:
    """Scale a mesh so that it fits inside the [-1, 1]^3 cube.

    The mesh is modified **in-place** and also returned for convenience.

    Args:
        mesh: Input mesh.
        center: Whether to translate the bounding-box center to the
            origin.

    Returns:
        A ``(mesh, params)`` tuple.  *mesh* is the same object
        (modified in-place) and *params* is a
        :class:`NormalizationParams` that can be passed to
        :func:`denormalize_mesh` to revert the transform.
    """
    bbox_center = None
    if center:
        bbox_min = mesh.vertices.min(axis=0)
        bbox_max = mesh.vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        mesh.vertices -= bbox_center

    extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    scale = extents.max()
    if scale > 0:
        mesh.vertices *= 2.0 / scale

    params = NormalizationParams(
        bbox_center=bbox_center,
        scale=float(scale),
    )
    return mesh, params


def denormalize_mesh(
    mesh: trimesh.Trimesh,
    params: NormalizationParams,
) -> trimesh.Trimesh:
    """Revert the normalization.

    Args:
        mesh: A mesh living in the normalized ``[-1, 1]^3`` space.
        params: The :class:`NormalizationParams` returned by
            :func:`normalize_mesh` on the *original* input mesh.

    Returns:
        The same mesh object, modified in-place.
    """
    if params.scale > 0:
        mesh.vertices *= params.scale / 2.0
    if params.bbox_center is not None:
        mesh.vertices += params.bbox_center

    mesh._cache.delete("face_normals")
    mesh._cache.delete("vertex_normals")

    return mesh


# ---------------------------------------------------------------------------
# Surface sampling
# ---------------------------------------------------------------------------


def sample_surface(
    mesh: trimesh.Trimesh,
    n_points: int,
    seed: int = 0,
    with_normals: bool = True,
    device: str | None = None,
    dtype: "torch.dtype | None" = None,
) -> torch.Tensor:
    """Sample *n_points* on the mesh surface and return a tensor.

    Points are sampled uniformly w.r.t. surface area.

    Args:
        mesh: Input mesh (must have faces).
        n_points: Number of points to sample.
        seed: Random seed for reproducibility.
        with_normals: If True, concatenate face normals to the
            positions, yielding shape ``(1, n_points, 6)``.
            Otherwise shape is ``(1, n_points, 3)``.

    Returns:
        Tensor of shape ``(1, n_points, 3|6)`` on the requested
        device / dtype.
    """
    points, face_indices = trimesh.sample.sample_surface(
        mesh,
        count=n_points,
        seed=seed,
    )
    surface = torch.from_numpy(np.asarray(points))
    if with_normals:
        normals = torch.from_numpy(
            np.asarray(mesh.face_normals[face_indices]),
        )
        surface = torch.cat([surface, normals], dim=-1)
    surface = surface.unsqueeze(0)
    if dtype is not None:
        surface = surface.to(dtype)
    if device is not None:
        surface = surface.to(device)
    return surface


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
