# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import subprocess
import sys

import numpy as np

logger = logging.getLogger(__name__)


def create_animated_glb(
    vertices_npy: str,
    faces_npy: str,
    output_glb: str,
    blender_path: str,
    fps: int = 24,
    export_normals: bool = False,
) -> int:
    """Create an animated GLB from vertex positions and faces.

    Launches Blender as a subprocess to create an animated mesh with shape keys.
    The mesh is created directly in Blender to avoid vertex reordering issues
    that occur when importing/exporting GLB files.

    Args:
        vertices_npy: Path to .npy file with vertex positions (N, V, 3)
            where N is number of timesteps, V is number of vertices.
        faces_npy: Path to .npy file with face indices (F, 3).
        output_glb: Path to save the output .glb file.
        blender_path: Path to Blender executable.
        fps: Frames per second for the animation. Default is 24.
        export_normals: Whether to export vertex normals. Defaults to False.

    Returns:
        Return code from Blender subprocess.
    """
    script_path = os.path.abspath(__file__)
    vertices_npy = os.path.abspath(vertices_npy)
    faces_npy = os.path.abspath(faces_npy)
    output_glb = os.path.abspath(output_glb)

    cmd = [
        blender_path,
        "-b",
        "-P",
        script_path,
        "--",
        "--vertices_npy",
        vertices_npy,
        "--faces_npy",
        faces_npy,
        "--output_glb",
        output_glb,
        "--fps",
        str(fps),
    ]
    if export_normals:
        cmd.append("--export_normals")

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode == 0:
        logger.info(f"Animated GLB saved to {output_glb}")
    else:
        logger.warning(
            f"Failed to save animated GLB (Blender exit code: {result.returncode})"
        )

    return result.returncode


def parse_args():
    """Parse command line arguments for Blender animation script."""
    parser = argparse.ArgumentParser(
        description="Blender animation script with shape keys from vertices and faces."
    )

    parser.add_argument(
        "--vertices_npy",
        type=str,
        required=True,
        help="Path to .npy file with vertex positions (N, V, 3)",
    )
    parser.add_argument(
        "--faces_npy",
        type=str,
        required=True,
        help="Path to .npy file with face indices (F, 3)",
    )
    parser.add_argument(
        "--output_glb",
        type=str,
        required=True,
        help="Path to save the output .glb file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second for the animation (default: 24)",
    )
    parser.add_argument(
        "--export_normals",
        action="store_true",
        help="Export vertex normals (default: False)",
    )

    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    else:
        parser.print_help()
        sys.exit(1)

    return args


def main():

    import bmesh
    import bpy
    from mathutils import Vector

    args = parse_args()

    # -- Load data
    vertices = np.load(args.vertices_npy)  # (N, V, 3)
    faces = np.load(args.faces_npy)  # (F, 3)
    num_frames, num_vertices, _ = vertices.shape

    # -- Clear the scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # -- Create mesh from first frame vertices and faces
    mesh = bpy.data.meshes.new("AnimatedMesh")
    obj = bpy.data.objects.new("AnimatedMesh", mesh)

    # Link object to scene
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Build mesh using bmesh for proper construction
    bm = bmesh.new()

    # Add vertices from first frame
    first_frame_verts = vertices[0]
    for v in first_frame_verts:
        bm.verts.new((v[0], v[1], v[2]))

    bm.verts.ensure_lookup_table()

    # Add faces
    for face in faces:
        try:
            bm.faces.new([bm.verts[int(i)] for i in face])
        except ValueError:
            # Face may already exist or be degenerate, skip
            pass

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # -- Create a nice blue material
    mat = bpy.data.materials.new(name="BlueMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create Principled BSDF shader
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    # Set a nice blue color (RGB)
    bsdf.inputs["Base Color"].default_value = (0.2, 0.4, 0.8, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.1
    bsdf.inputs["Roughness"].default_value = 0.4

    # Create Material Output node
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)

    # Link BSDF to Material Output
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Assign material to mesh
    obj.data.materials.append(mat)

    # -- Add Shape Keys for animation
    obj.shape_key_add(name="Basis")  # Basis shape key (required)

    # Create shape keys for each frame
    for frame_idx in range(num_frames):
        shape_key = obj.shape_key_add(name=f"Frame_{frame_idx}")

        # Apply vertex positions for this frame
        frame_verts = vertices[frame_idx]
        for vert_idx in range(num_vertices):
            shape_key.data[vert_idx].co = Vector(
                (
                    frame_verts[vert_idx, 0],
                    frame_verts[vert_idx, 1],
                    frame_verts[vert_idx, 2],
                )
            )

        # Animate shape key influence
        shape_key.value = 1.0
        shape_key.keyframe_insert(data_path="value", frame=frame_idx)
        if frame_idx > 0:
            shape_key.value = 0.0
            shape_key.keyframe_insert(data_path="value", frame=frame_idx - 1)
        if frame_idx < num_frames - 1:
            shape_key.value = 0.0
            shape_key.keyframe_insert(data_path="value", frame=frame_idx + 1)

    # Adjust timeline frame range and FPS
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    bpy.context.scene.render.fps = args.fps

    # Export GLB with Draco compression
    bpy.ops.export_scene.gltf(
        filepath=args.output_glb,
        export_format="GLB",
        export_texcoords=False,
        export_materials="EXPORT",
        export_optimize_animation_size=True,
        export_normals=args.export_normals,
        export_tangents=False,
        export_morph_normal=False,
        export_morph_tangent=False,
        # Draco mesh compression
        export_draco_mesh_compression_enable=True,
        export_draco_mesh_compression_level=6,
        export_draco_position_quantization=14,
        export_draco_normal_quantization=10,
    )


if __name__ == "__main__":
    main()
