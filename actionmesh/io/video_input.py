# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from natsort import natsorted
from PIL import Image

logger = logging.getLogger(__name__)

# Video formats reliably supported by OpenCV on most systems
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
# Image formats supported by PIL/Pillow
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Minimum number of frames required for ActionMesh
MIN_FRAMES = 16


@dataclass
class ActionMeshInput:
    """
    Input data for ActionMesh pipeline.

    Attributes:
        frames: List of N input video frames as RGB or RGBA PIL Images.
        timesteps: Tensor of shape (N,) containing video timesteps [0, 1, 2, ...], float32 on CPU.
    """

    frames: list[Image.Image]
    timesteps: torch.Tensor

    def __post_init__(self) -> None:
        assert (
            len(self.frames) >= MIN_FRAMES
        ), f"At least {MIN_FRAMES} frames are required, got {len(self.frames)}"
        assert (
            self.timesteps.ndim == 1
        ), f"Expected 1D timesteps, got {self.timesteps.ndim}D"
        assert (
            len(self.frames) == self.timesteps.shape[0]
        ), f"Number of frames ({len(self.frames)}) must match timesteps ({self.timesteps.shape[0]})"
        assert (
            self.timesteps.dtype == torch.float32
        ), f"Expected float32 timesteps, got {self.timesteps.dtype}"
        assert (
            self.timesteps.device.type == "cpu"
        ), f"Expected CPU timesteps, got {self.timesteps.device}"

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def get(self, indices: torch.Tensor | list[int]) -> "ActionMeshInput":
        """
        Return a new ActionMeshInput with only the selected indices.
        """
        indices_list = (
            indices.tolist() if isinstance(indices, torch.Tensor) else indices
        )
        return ActionMeshInput(
            frames=[self.frames[i] for i in indices_list],
            timesteps=self.timesteps[indices],
        )


def load_from_image_mask_pairs(
    directory: str | Path, max_frames: int | None = None, stride: int = 1
) -> ActionMeshInput:
    """
    Load frames from a directory containing separate image and mask files.

    Expects files named as xxx_image.png and xxx_mask.png pairs.
    Combines each image with its corresponding mask into an RGBA image.

    Args:
        directory: Path to directory containing *_image.png and *_mask.png files.
        max_frames: Maximum number of frames to load. None for all frames.
        stride: Take every nth frame (default=1, i.e., all frames).

    Returns:
        ActionMeshInput with loaded RGBA frames and sequential timesteps [0, 1, 2, ...].
    """
    directory = Path(directory)
    image_files = sorted(directory.glob("*_image.png"))

    if not image_files:
        raise ValueError(f"No *_image.png files found in '{directory}'")

    # Apply stride first, then max_frames
    image_files = image_files[::stride]
    if max_frames is not None:
        image_files = image_files[:max_frames]

    frames = []
    for image_file in image_files:
        prefix = image_file.stem.replace("_image", "")
        mask_file = directory / f"{prefix}_mask.png"

        if not mask_file.exists():
            raise ValueError(
                f"No mask found for {image_file.name}: expected {mask_file}"
            )

        image = Image.open(image_file).convert("RGB")
        mask = Image.open(mask_file).convert("L")

        if image.size != mask.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        rgba = image.copy()
        rgba.putalpha(mask)
        frames.append(rgba)

    timesteps = torch.arange(len(frames), dtype=torch.float32)

    logger.info(f"Loaded {len(frames)} frames from image+mask pairs: {directory}")
    return ActionMeshInput(frames=frames, timesteps=timesteps)


def load_from_image_dir(
    path_pattern: str | Path, max_frames: int | None = None, stride: int = 1
) -> ActionMeshInput:
    """
    Load frames from a directory matching a glob pattern.

    Args:
        path_pattern: Path with glob pattern (e.g., "/path/to/frames/*.png").
        max_frames: Maximum number of frames to load. None for all frames.
        stride: Take every nth frame (default=1, i.e., all frames).

    Returns:
        ActionMeshInput with loaded frames and sequential timesteps [0, 1, 2, ...].
    """
    path_pattern = Path(path_pattern)
    image_paths = natsorted(path_pattern.parent.glob(path_pattern.name))

    if not image_paths:
        raise ValueError(f"No images found matching '{path_pattern}'")

    # Apply stride first, then max_frames
    image_paths = image_paths[::stride]
    if max_frames is not None:
        image_paths = image_paths[:max_frames]

    frames = [Image.open(path).convert("RGBA") for path in image_paths]
    timesteps = torch.arange(len(frames), dtype=torch.float32)

    logger.info(f"Loaded {len(frames)} frames from image folder: {path_pattern.parent}")
    return ActionMeshInput(frames=frames, timesteps=timesteps)


def load_from_video(
    video_path: str | Path, max_frames: int | None = None, stride: int = 1
) -> ActionMeshInput:
    """
    Load frames from a video file using OpenCV.

    Args:
        video_path: Path to video file (e.g., .mp4, .avi, .mov).
        max_frames: Maximum number of frames to load. None for all frames.
        stride: Take every nth frame (default=1, i.e., all frames).

    Returns:
        ActionMeshInput with loaded frames and sequential timesteps [0, 1, 2, ...].
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        frames = []
        frame_idx = 0
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Only keep every nth frame based on stride
            if frame_idx % stride == 0:
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frames.append(Image.fromarray(frame_rgba))
            frame_idx += 1
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")

    timesteps = torch.arange(len(frames), dtype=torch.float32)

    logger.info(f"Loaded {len(frames)} frames from video: {video_path}")
    return ActionMeshInput(frames=frames, timesteps=timesteps)


def load_frames(
    path: str | Path, max_frames: int | None = None, stride: int = 1
) -> ActionMeshInput:
    """
    Load frames from either a video file or image directory pattern.

    Automatically detects the input type:
    - If path has a video extension (.mp4, .avi, etc.), loads as video
    - If path contains glob characters (* or ?), loads as image pattern
    - If path is a directory, loads all images from it

    Args:
        path: Path to video file, glob pattern, or directory.
        max_frames: Maximum number of frames to load. None for all frames.
        stride: Take every nth frame (default=1, i.e., all frames).

    Returns:
        ActionMeshInput with loaded frames and sequential timesteps [0, 1, 2, ...].
    """
    path = Path(path)
    path_str = str(path)

    # Check for glob pattern
    if "*" in path_str or "?" in path_str:
        return load_from_image_dir(path, max_frames=max_frames, stride=stride)

    # Check for video file
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return load_from_video(path, max_frames=max_frames, stride=stride)

    # Check for directory
    if path.is_dir():
        # Check for image/mask pairs first (xxx_image.png + xxx_mask.png)
        image_mask_files = list(path.glob("*_mask.png"))
        if image_mask_files:
            return load_from_image_mask_pairs(
                path, max_frames=max_frames, stride=stride
            )

        # Try common image patterns
        for ext in IMAGE_EXTENSIONS:
            pattern = path / f"*{ext}"
            try:
                return load_from_image_dir(
                    pattern, max_frames=max_frames, stride=stride
                )
            except ValueError:
                continue
        raise ValueError(f"No images found in directory: {path}")

    raise ValueError(
        f"Unsupported input: {path}. Expected video file, image pattern, or directory."
    )
