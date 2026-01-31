# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from transformers import BitImageProcessor, Dinov2Model


@dataclass(eq=False)
class ImageEncoder(nn.Module):
    pretrained_dino_feature_extractor: str
    pretrained_dino_model: str

    def __post_init__(self):
        super().__init__()

        # -- Load the DINOv2 model + image processor
        self.dino_model: Dinov2Model = Dinov2Model.from_pretrained(
            self.pretrained_dino_model
        )
        self.dino_model.eval()

        self.image_preprocess_dino = BitImageProcessor.from_pretrained(
            self.pretrained_dino_feature_extractor
        )

    @property
    def device(self) -> torch.device:
        return next(self.dino_model.parameters()).device

    def encode_images(
        self,
        images: List[Image.Image],
    ) -> torch.FloatTensor:
        """
        Args:
            images: List of T PIL images to encode.
        Returns:
            context (T, S, Dc): The context embeddings for the given images.
        """
        pixel_values = self.image_preprocess_dino.preprocess(
            images,
            return_tensors="pt",
        ).pixel_values

        vision_outputs = self.dino_model(pixel_values.to(self.dino_model.device))

        return vision_outputs.last_hidden_state
