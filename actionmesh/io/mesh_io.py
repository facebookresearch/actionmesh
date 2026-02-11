# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def load_glb(path: str) -> trimesh.Trimesh:
    """Load a .glb file and return a single trimesh.Trimesh.

    If the file contains a scene with multiple geometries they are
    concatenated into one mesh.

    Args:
        path: Path to the .glb file.

    Returns:
        A single trimesh.Trimesh object.

    Raises:
        ValueError: If the file contains no geometry.
    """
    loaded = trimesh.load(str(path), file_type="glb", force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geo for geo in loaded.geometry.values() if isinstance(geo, trimesh.Trimesh)
        ]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        loaded = trimesh.util.concatenate(meshes)
    return loaded


def save_deformation(
    meshes: list[trimesh.Trimesh],
    path: str | Path,
) -> tuple[Path, Path]:
    """
    Save vertex positions and faces from a list of meshes to .npy files.

    Saves:
        - vertices.npy: A numpy array of shape (T, N_Verts, 3) with dtype float32.
        - faces.npy: A numpy array of shape (N_Faces, 3) with dtype int32.

    Args:
        meshes: List of trimesh meshes with consistent topology.
        path: Base path for output files. Will create {path}_vertices.npy and {path}_faces.npy

    Returns:
        Tuple of (vertices_path, faces_path).

    Raises:
        ValueError: If meshes list is empty, vertex counts don't match,
            or face topologies differ between meshes.
    """
    if len(meshes) == 0:
        raise ValueError("Cannot save deformation from empty mesh list")

    n_verts = len(meshes[0].vertices)
    reference_faces = meshes[0].faces

    for i, mesh in enumerate(meshes):
        if len(mesh.vertices) != n_verts:
            raise ValueError(
                f"Mesh {i} has {len(mesh.vertices)} vertices, "
                f"expected {n_verts} (same as first mesh)"
            )
        if mesh.faces.shape != reference_faces.shape or not np.array_equal(
            mesh.faces, reference_faces
        ):
            raise ValueError(
                f"Mesh {i} has different face topology than the first mesh. "
                f"All meshes must share the same faces for deformation export."
            )

    vertices = np.stack(
        [mesh.vertices.astype(np.float32) for mesh in meshes],
        axis=0,
    )
    vertices = vertices[:, :, [2, 0, 1]]
    vertices[:, :, 0] = -vertices[:, :, 0]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save vertices and faces as separate files
    vertices_path = path.parent / f"{path.stem}_vertices.npy"
    faces_path = path.parent / f"{path.stem}_faces.npy"

    np.save(vertices_path, vertices)
    np.save(faces_path, reference_faces.astype(np.int32))

    return vertices_path, faces_path


def save_meshes(meshes: list[trimesh.Trimesh], output_dir: str):
    """
    Save meshes in output directory.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for i in range(len(meshes)):
        meshes[i].export(f"{output_dir}/mesh_{i:02d}.glb")

    logger.info(f"Saved {len(meshes)} meshes to {output_dir}")
