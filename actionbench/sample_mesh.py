# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import trimesh
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.structures import join_meshes_as_batch, Meshes


def trimesh_to_pytorch3d(mesh: trimesh.Trimesh, device: str = "cpu") -> Meshes:
    """
    Convert a trimesh mesh to a pytorch3d Meshes object.

    Args:
        mesh: Input trimesh object.
        device: Device to place tensors on.

    Returns:
        PyTorch3D Meshes object.
    """
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return Meshes(verts=[verts], faces=[faces])


def _rand_barycentric_coords(
    size1: int,
    size2: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate random barycentric coordinates uniformly over a triangle.

    Args:
        size1: First dimension of output tensors.
        size2: Second dimension of output tensors.
        dtype: Data type for the output tensors.
        device: Device to place tensors on.

    Returns:
        Tuple of (w0, w1, w2) tensors of shape (size1, size2).
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2


def get_baryc_sampling_mesh(
    mesh: Meshes,
    num_samples: int,
    seed: int = 44,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute face indices and barycentric coordinates for sampling a mesh.

    Args:
        mesh: Single PyTorch3D Meshes object.
        num_samples: Number of points to sample.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sample_face_idxs, baryc_coords) tensors.
    """
    if mesh.isempty():
        raise ValueError("Meshes are empty.")

    verts = mesh.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = mesh.faces_packed()
    mesh_to_face = mesh.mesh_to_faces_packed_first_idx()
    num_meshes = len(mesh)
    num_valid_meshes = torch.sum(mesh.valid)

    assert num_meshes == 1, "Only one mesh is supported."

    with torch.no_grad():
        with torch.random.fork_rng(devices=[verts.device], enabled=True):
            torch.manual_seed(seed)
            areas, _ = mesh_face_areas_normals(verts, faces)
            max_faces = mesh.num_faces_per_mesh().max().item()
            areas_padded = packed_to_padded(areas, mesh_to_face[mesh.valid], max_faces)

            sample_face_idxs = areas_padded.multinomial(num_samples, replacement=True)
            offset = mesh_to_face[mesh.valid].view(num_valid_meshes, 1)
            sample_face_idxs += offset

            w0, w1, w2 = _rand_barycentric_coords(
                num_valid_meshes, num_samples, verts.dtype, verts.device
            )
            baryc_coords = torch.stack([w0, w1, w2], dim=-1)

    return sample_face_idxs, baryc_coords


def apply_baryc_sampling_on_meshes(
    meshes: Meshes,
    sample_face_idxs: torch.Tensor,
    baryc_coords: torch.Tensor,
) -> torch.Tensor:
    """
    Apply precomputed barycentric sampling to a batch of meshes.

    Args:
        meshes: Batch of PyTorch3D Meshes with same topology.
        sample_face_idxs: (1, num_samples) face indices from reference mesh.
        baryc_coords: (1, num_samples, 3) barycentric coordinates.

    Returns:
        (num_meshes, num_samples, 3) sampled points.
    """
    assert all(
        torch.all(meshes.faces_list()[0] == face_list)
        for face_list in meshes.faces_list()
    )

    num_meshes = len(meshes)
    num_samples = sample_face_idxs.shape[-1]
    w0, w1, w2 = torch.unbind(baryc_coords, dim=-1)

    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    for k in range(num_meshes):
        if not meshes[k].valid:
            continue

        verts_packed = meshes[k].verts_packed()
        faces_packed = meshes[k].faces_packed()
        face_verts = verts_packed[faces_packed]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        a = v0[sample_face_idxs]
        b = v1[sample_face_idxs]
        c = v2[sample_face_idxs]

        samples[k] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    return samples


def sample_synchronized_points(
    meshes: list[trimesh.Trimesh],
    n_pts: int,
    seed: int = 44,
    root_idx: int = 0,
) -> torch.Tensor:
    """
    Sample points from meshes using synchronized barycentric coordinates.

    Computes sampling from the root mesh and applies it to all meshes,
    ensuring point correspondence across the sequence.

    Args:
        meshes: List of trimesh objects with same topology.
        n_pts: Number of points to sample.
        seed: Random seed for reproducibility.
        root_idx: Index of the mesh to use for computing sampling.

    Returns:
        (T, n_pts, 3) sampled points for each mesh.
    """
    sample_face_idxs, baryc_coords = get_baryc_sampling_mesh(
        mesh=trimesh_to_pytorch3d(meshes[root_idx]),
        num_samples=n_pts,
        seed=seed,
    )

    samples = apply_baryc_sampling_on_meshes(
        meshes=join_meshes_as_batch([trimesh_to_pytorch3d(mesh) for mesh in meshes]),
        sample_face_idxs=sample_face_idxs,
        baryc_coords=baryc_coords,
    )

    return samples


def sample_points(
    mesh: trimesh.Trimesh,
    n_pts: int,
    seed: int = 44,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample points uniformly from mesh surface.

    Args:
        mesh: Input trimesh object.
        n_pts: Number of points to sample.
        seed: Random seed for reproducibility.
        device: Device to place output tensor on.

    Returns:
        (n_pts, 3) sampled points.
    """
    sampled_points, _ = trimesh.sample.sample_surface(
        mesh,
        count=n_pts,
        seed=seed,
    )
    return torch.from_numpy(sampled_points).float().to(device)


def sample_meshes(
    meshes: list[trimesh.Trimesh],
    n_pts: int = 100_000,
    synchronized: bool = False,
    seed: int = 44,
) -> torch.Tensor:
    """
    Sample points from a sequence of meshes.

    Args:
        meshes: List of trimesh objects.
        n_pts: Number of points to sample per mesh.
        synchronized: If True, use synchronized sampling for correspondence.
        seed: Random seed for reproducibility.

    Returns:
        (T, n_pts, 3) sampled points for each mesh.
    """
    if synchronized:
        return sample_synchronized_points(
            meshes=meshes,
            n_pts=n_pts,
            seed=seed,
            root_idx=0,
        )

    samples = [
        sample_points(mesh=mesh, n_pts=n_pts, seed=seed + i)
        for i, mesh in enumerate(meshes)
    ]
    return torch.stack(samples)
