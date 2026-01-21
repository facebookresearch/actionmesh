# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path

import torch
from actionmesh.io.glb_export import create_animated_glb
from actionmesh.io.mesh_io import save_deformation, save_meshes
from actionmesh.io.video_input import load_frames
from actionmesh.pipeline import ActionMeshPipeline

# Configure logging for actionmesh modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_pytorch3d_installed() -> bool:
    """Check if pytorch3d is installed."""
    try:
        import pytorch3d

        return True
    except ImportError:
        logger.warning(
            "PyTorch3D is not installed. Video rendering will be skipped. "
            "See https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
        )
        return False


def check_blender_available(blender_path: str | None = None) -> bool:
    """Check if Blender is available and return the path to the executable."""
    if blender_path is None:
        logger.warning(
            "No Blender path provided. animated_mesh.glb will not be saved. "
            "Use --blender_path to specify your Blender 3.5.1 executable."
        )
        return False

    if os.path.isfile(blender_path) and os.access(blender_path, os.X_OK):
        return True
    else:
        logger.warning(
            f"Provided Blender path '{blender_path}' is not a valid executable. "
            "animated_mesh.glb will not be saved."
        )
        return False


@torch.no_grad()
def run_actionmesh(
    pipeline: ActionMeshPipeline,
    input: torch.Tensor,
    output_dir: str,
    seed: int,
    blender_path: str | None = None,
    # -- Pipeline parameters
    stage_0_steps: int | None = None,
    face_decimation: int | None = None,
    floaters_threshold: float | None = None,
    stage_1_steps: int | None = None,
    guidance_scales: list[float] | None = None,
    anchor_idx: int | None = None,
):

    # -- Prepare input video
    input = load_frames(path=input, max_frames=31)

    # -- Run inference
    meshes = pipeline(
        input=input,
        seed=seed,
        stage_0_steps=stage_0_steps,
        face_decimation=face_decimation,
        floaters_threshold=floaters_threshold,
        stage_1_steps=stage_1_steps,
        guidance_scales=guidance_scales,
        anchor_idx=anchor_idx,
    )

    # -- Save meshes + T vertices + faces
    save_meshes(meshes, output_dir=output_dir)
    vertices_path, faces_path = save_deformation(
        meshes, path=f"{output_dir}/deformations"
    )

    # -- [Optional] Create animated GLB (requires Blender 3.5.1)
    if check_blender_available(blender_path):
        animated_glb_path = f"{output_dir}/animated_mesh.glb"
        create_animated_glb(
            blender_path=blender_path,
            vertices_npy=vertices_path,
            faces_npy=faces_path,
            output_glb=animated_glb_path,
        )

    # -- [Optional] Render output (automatically if pytorch3d is installed)
    if check_pytorch3d_installed():
        from actionmesh.render.visualizer import ActionMeshVisualizer

        visualizer = ActionMeshVisualizer(image_size=256)
        visualizer.render(
            meshes,
            input_frames=input.frames,
            device=pipeline.device,
            output_dir=output_dir,
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video (.mp4) or folder containing PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated meshes. Default: outputs/<input_name>",
    )
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--blender_path",
        type=str,
        default=None,
        help="Path to Blender executable.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast preset (stage_0_steps=50, stage_1_steps=15).",
    )
    # -- Pipeline parameters
    parser.add_argument(
        "--stage_0_steps",
        type=int,
        default=None,
        help="Number of inference steps for image-to-3D (TripoSG). Default: 100. Fast: 50",
    )
    parser.add_argument(
        "--face_decimation",
        type=int,
        default=None,
        help="Target number of faces for mesh decimation. Default: 40000",
    )
    parser.add_argument(
        "--floaters_threshold",
        type=float,
        default=None,
        help="Threshold for removing floaters (0.0-1.0). Default: 0.02",
    )
    parser.add_argument(
        "--stage_1_steps",
        type=int,
        default=None,
        help="Number of flow-matching denoising steps in ActionMesh temporal 3D denoiser (Stage I). Default: 30. Fast: 15",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=None,
        help="Classifier-free guidance scales in ActionMesh temporal 3D denoiser. Default: [7.5]",
    )
    parser.add_argument(
        "--anchor_idx",
        type=int,
        default=None,
        help="Index of the anchor frame (fixing the topology). Default: 0",
    )
    args = parser.parse_args()

    # -- Apply fast preset if requested
    if args.fast:
        # Check that no pipeline parameters are set when --fast is enabled
        fast_incompatible_params = [
            ("--stage_0_steps", args.stage_0_steps),
            ("--stage_1_steps", args.stage_1_steps),
            ("--face_decimation", args.face_decimation),
            ("--floaters_threshold", args.floaters_threshold),
            ("--guidance_scales", args.guidance_scales),
            ("--anchor_idx", args.anchor_idx),
        ]
        for param_name, param_value in fast_incompatible_params:
            if param_value is not None:
                raise ValueError(
                    f"{param_name} cannot be set when --fast is enabled. "
                    "Remove --fast to use custom parameters."
                )
        logger.info(
            "Fast mode enabled: quality might be slightly reduced compared to default settings."
        )
        args.stage_0_steps = 50
        args.stage_1_steps = 15
        args.face_decimation = 40000
        args.floaters_threshold = 0.02
        args.guidance_scales = [7.5]

    # -- Set default output directory if not provided
    if args.output_dir is None:
        input_name = Path(args.input).stem
        args.output_dir = f"outputs/{input_name}"
        logger.info(f"Output directory not specified, using: {args.output_dir}")

    # -- Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -- Initialize ActionMesh pipeline
    device = "cuda"
    config_dir = Path(__file__).parent.parent / "actionmesh" / "configs"
    pipeline: ActionMeshPipeline = ActionMeshPipeline(
        config_name="actionmesh.yaml",
        config_dir=str(config_dir),
    )
    pipeline.to(device)

    # -- Run ActionMesh inference
    run_actionmesh(
        pipeline,
        input=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        blender_path=args.blender_path,
        stage_0_steps=args.stage_0_steps,
        face_decimation=args.face_decimation,
        floaters_threshold=args.floaters_threshold,
        stage_1_steps=args.stage_1_steps,
        guidance_scales=args.guidance_scales,
        anchor_idx=args.anchor_idx,
    )
