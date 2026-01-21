# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def is_valid_alpha(
    alpha: np.ndarray, min_ratio: float = 0.01, threshold: int = 127
) -> bool:
    """Check if alpha channel has sufficient foreground and background pixels."""
    total_pixels = alpha.size
    min_count = int(total_pixels * min_ratio)
    fg_count = np.count_nonzero(alpha > threshold)
    bg_count = total_pixels - fg_count
    return bg_count >= min_count and fg_count >= min_count


def load_image(
    image: Image.Image,
    bg_color: np.ndarray,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Load and process an RGBA image, applying background color."""
    # Ensure RGBA format
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Get image as contiguous numpy array
    img_array = np.ascontiguousarray(image)
    rgb = img_array[..., :3]
    alpha = img_array[..., 3]

    if not is_valid_alpha(alpha):
        raise ValueError("Invalid alpha channel: insufficient foreground/background")

    # Normalize alpha to [0, 1] as float32 for compositing
    alpha_norm = alpha.astype(np.float32) * (1.0 / 255.0)

    # Composite RGB with background
    # rgb_out = rgb * alpha + bg * (1 - alpha)
    alpha_3ch = alpha_norm[..., np.newaxis]  # (H, W, 1)
    bg_color_f32 = bg_color.astype(np.float32)
    rgb_composite = rgb.astype(np.float32) * (
        1.0 / 255.0
    ) * alpha_3ch + bg_color_f32 * (1.0 - alpha_3ch)

    # Convert to tensor (HWC -> CHW)
    rgb_pt = torch.from_numpy(np.ascontiguousarray(rgb_composite.transpose(2, 0, 1)))

    # Compute bounding box from alpha mask
    alpha_mask = alpha > 0
    rows_any = np.any(alpha_mask, axis=1)
    cols_any = np.any(alpha_mask, axis=0)
    row_indices = np.nonzero(rows_any)[0]
    col_indices = np.nonzero(cols_any)[0]
    y, y_max = row_indices[0], row_indices[-1]
    x, x_max = col_indices[0], col_indices[-1]
    w, h = x_max - x + 1, y_max - y + 1

    return rgb_pt, (x, y, w, h)


def aggregate_bboxes(
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    """Compute the bounding box that encompasses all input bboxes."""
    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[0] + b[2] for b in bboxes)
    y_max = max(b[1] + b[3] for b in bboxes)
    return x_min, y_min, x_max - x_min, y_max - y_min


def apply_padding(
    rgb_image: torch.Tensor,
    bbox: tuple[int, int, int, int],
    padding_ratio: float = 0.1,
    padding_value: float = 1.0,
) -> torch.Tensor:
    """Crop to bounding box and pad to make square with additional margin."""
    x, y, w, h = bbox

    # Crop to bounding box
    cropped = rgb_image[:, y : y + h, x : x + w]

    # Calculate padding to make square with margin
    max_dim = max(w, h)
    pad_base = int(max_dim * padding_ratio)
    pad_x = pad_base + (max_dim - w) // 2
    pad_y = pad_base + (max_dim - h) // 2

    return F.pad(
        cropped, (pad_x, pad_x, pad_y, pad_y), mode="constant", value=padding_value
    )


@dataclass(eq=False)
class ImagePreprocessor:
    """Preprocessor for ActionMesh input images.

    Handles RGBA images by compositing onto a white background, cropping to
    the foreground region, and padding to create square outputs.

    Attributes:
        independent_cropping: If True, crop each frame independently. If False,
            use a shared bounding box across all frames for consistent framing.
        padding_ratio: Padding to add around the cropped region as a fraction
            of the largest dimension.
    """

    independent_cropping: bool = False
    padding_ratio: float = 0.1

    def __post_init__(self):
        self.bg_color = np.array([1.0, 1.0, 1.0])

    def process_images(self, frames: list[Image.Image]) -> list[Image.Image]:
        """Process frames: load, crop, pad, and return as PIL images."""

        results = [load_image(frame, self.bg_color) for frame in frames]
        images = [r[0] for r in results]
        bboxes = [r[1] for r in results]

        # Use aggregated bbox if not cropping independently
        if not self.independent_cropping:
            agg_bbox = aggregate_bboxes(bboxes)
            bboxes = [agg_bbox] * len(bboxes)

        # Crop and pad each image
        processed = [
            apply_padding(img, bbox, self.padding_ratio, self.bg_color[0])
            for img, bbox in zip(images, bboxes)
        ]

        # Convert back to PIL images
        return [
            Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            for img in processed
        ]
