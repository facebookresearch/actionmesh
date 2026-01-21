# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Rendering utilities for video and image output."""

from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def resample_list(items: list, target_length: int) -> list:
    """
    Resample a list to a target length using nearest-neighbor interpolation.

    Args:
        items: Input list of any elements.
        target_length: Desired output length.

    Returns:
        Resampled list of length target_length.
    """
    if not items or target_length <= 0:
        return []
    n_in = len(items)
    if target_length == 1:
        return [items[0]]

    return [
        items[round(i * (n_in - 1) / (target_length - 1) + 1e-4)]
        for i in range(target_length)
    ]


def make_image_grid(
    images: list[Image.Image],
    n_cols: int,
    image_size: int | None = None,
) -> Image.Image:
    """
    Create a grid image from a list of images.

    Args:
        images: List of PIL Images.
        n_cols: Number of columns in the grid.
        image_size: Optional size to resize images to (square).

    Returns:
        Single PIL Image containing the grid.
    """
    if image_size is not None:
        images = [img.resize((image_size, image_size)) for img in images]

    n_rows = (len(images) + n_cols - 1) // n_cols
    w, h = images[0].size
    grid = Image.new("RGBA", (n_cols * w, n_rows * h), (0, 0, 0, 0))

    for idx, img in enumerate(images):
        col, row = idx % n_cols, idx // n_cols
        grid.paste(img, (col * w, row * h))

    return grid


def save_video(
    frames: list[Image.Image],
    output_path: str | Path,
    fps: int = 12,
) -> None:
    """
    Save a list of PIL Images as an MP4 video.

    Args:
        frames: List of PIL Images.
        output_path: Path to save the video.
        fps: Frames per second.
    """
    if not frames:
        return

    frames_rgb = [np.array(frame.convert("RGB")) for frame in frames]
    imageio.mimsave(str(output_path), frames_rgb, fps=fps)


def save_rgba_video(
    frames: list[Image.Image],
    output_path: str | Path,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    fps: int = 12,
) -> None:
    """
    Save RGBA frames as a video, compositing onto a solid background.

    Args:
        frames: List of PIL Images (RGBA mode).
        output_path: Path to save the video.
        bg_color: RGB background color (0-255). Defaults to white.
        fps: Frames per second.
    """
    if not frames:
        return

    composited = []
    for frame in frames:
        rgba = frame.convert("RGBA")
        background = Image.new("RGB", rgba.size, bg_color)
        background.paste(rgba, mask=rgba.split()[3])
        composited.append(background)

    save_video(composited, output_path, fps=fps)


def save_multiview_video_grid(
    views: list[list[dict[str, Image.Image]]],
    output_dir: str | Path,
    modalities: list[str] | None = None,
    n_cols: int = 2,
    fps: int = 12,
    image_size: int | None = None,
) -> list[Path]:
    """
    Save multi-view predictions as grid videos, one per modality.

    Args:
        views: List of views, where each view is a list of frames,
            and each frame is a dict mapping modality name to PIL Image.
        output_dir: Directory to save videos.
        modalities: Modalities to save. If None, saves all available.
        n_cols: Number of columns in the grid.
        fps: Frames per second.
        image_size: Optional size to resize images to (square).

    Returns:
        List of saved file paths.
    """
    if not views or not views[0]:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_frames = len(views[0])

    # Find modalities available in all views and frames
    available = set(views[0][0].keys())
    for view in views:
        for frame in view:
            available &= set(frame.keys())

    # Filter to requested modalities
    if modalities is not None:
        modalities = [m for m in modalities if m in available]
    else:
        modalities = sorted(available)

    saved_files = []
    for modality in modalities:
        grid_frames = [
            make_image_grid(
                [view[i][modality] for view in views],
                n_cols,
                image_size,
            )
            for i in range(n_frames)
        ]
        output_path = output_dir / f"grid_{modality}.mp4"
        save_video(grid_frames, output_path, fps=fps)
        saved_files.append(output_path)

    return saved_files
