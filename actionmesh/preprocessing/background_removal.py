# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from actionmesh.preprocessing.image_processor import is_valid_alpha
from PIL import Image
from skimage.measure import label
from skimage.morphology import remove_small_objects
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation


def refine_mask(mask: np.ndarray, min_size: int = 200) -> np.ndarray:
    """Refine a soft mask into a clean binary mask.

    Applies Otsu's thresholding to binarize the mask, then removes small
    connected components to clean up noise.

    Args:
        mask: Grayscale mask as numpy array (H, W) with values 0-255.
        min_size: Minimum size of connected components to keep.

    Returns:
        Binary mask as numpy array (H, W) with values 0 or 255.
    """
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled_mask = label(binary_mask)
    cleaned_mask = remove_small_objects(labeled_mask, min_size=min_size)
    cleaned_mask = (cleaned_mask > 0).astype(np.uint8) * 255

    return cleaned_mask


class BackgroundRemover(nn.Module):
    def __init__(
        self,
        rmbg_weights_dir: str = "briaai/RMBG-1.4",
        model_input_size: tuple[int, int] = (1024, 1024),
    ):
        super().__init__()
        self.model = AutoModelForImageSegmentation.from_pretrained(
            rmbg_weights_dir, trust_remote_code=True
        )
        self.model_input_size = model_input_size

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _preprocess_image(self, im: np.ndarray) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0),
            size=self.model_input_size,
            mode="bilinear",
            align_corners=False,
        )
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return image

    def _postprocess_mask(
        self, result: torch.Tensor, im_size: tuple[int, int]
    ) -> np.ndarray:
        result = torch.squeeze(
            F.interpolate(result, size=im_size, mode="bilinear", align_corners=False), 0
        )
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array

    @torch.no_grad()
    def forward(
        self, image: np.ndarray, refine: bool = True, min_size: int = 200
    ) -> np.ndarray:
        """
        Remove background from an RGB image.

        Args:
            image: RGB image as numpy array with shape (H, W, 3) and dtype uint8.
            refine: Whether to apply mask refinement (Otsu + small object removal).
            min_size: Minimum size of connected components to keep when refining.

        Returns:
            RGBA image as numpy array with shape (H, W, 4) and dtype uint8,
            where the alpha channel is the foreground mask.
        """
        orig_im_size = image.shape[:2]

        preprocessed = self._preprocess_image(image).to(self.device)

        result = self.model(preprocessed)

        mask = self._postprocess_mask(result[0][0], orig_im_size)

        if refine:
            mask = refine_mask(mask, min_size=min_size)

        rgba = np.concatenate([image, mask[:, :, np.newaxis]], axis=2)
        return rgba

    def _has_a_valid_alpha_mask(self, image: Image.Image, threshold: int = 127) -> bool:
        """Check if image has a valid alpha mask using histogram analysis.

        Args:
            image: PIL Image to check.
            threshold: Threshold value to binarize the alpha mask before validation.

        Returns:
            True if the image has a valid binary alpha mask.
        """
        if image.mode != "RGBA":
            return False
        alpha = np.array(image.split()[3])
        binary_alpha = np.where(alpha > threshold, 255, 0).astype(np.uint8)
        return is_valid_alpha(binary_alpha)

    def process_image(self, image: Image.Image) -> Image.Image:
        """
        Remove background from a PIL RGB image.

        Args:
            image: PIL Image in RGB or RGBA mode.

        Returns:
            PIL Image in RGBA mode with background removed.
            If image already has a valid alpha mask, returns it as-is.
        """
        if self._has_a_valid_alpha_mask(image):
            return image
        rgb_array = np.array(image.convert("RGB"))
        rgba_array = self.forward(rgb_array)
        return Image.fromarray(rgba_array, mode="RGBA")

    def process_images(self, images: list[Image.Image]) -> list[Image.Image]:
        return [self.process_image(image) for image in images]
