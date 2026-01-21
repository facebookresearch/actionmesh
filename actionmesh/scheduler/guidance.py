# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(eq=False)
class ClassifierFreeGuidance:
    """
    Assume conditioning order is [image-conditioning | latent0 conditioning]
    """

    inference_enabled: bool = True
    guidance_at_inference: list[int] = field(
        default_factory=lambda: [[0, 0], [0, 1], [1, 1]]
    )
    guidance_scales: list[float] = field(default_factory=lambda: [1.0, 1.0])

    def __post_init__(self):
        assert len(self.guidance_at_inference) == len(self.guidance_scales) + 1

    def get_unobserved_mask(
        self, mask: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Returns a mask of the unobserved points
        """
        if mask is None:
            return None
        return mask == 0

    def cfg_at_inference(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        framestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate the batch which contains all the elements to perform Classifier-Free Guidance.
        """
        if not self.inference_enabled:
            return (
                latent,
                context,
                mask,
                framestep,
            )

        latent = torch.cat([latent] * len(self.guidance_at_inference))

        framestep = (
            torch.cat([framestep] * len(self.guidance_at_inference))
            if framestep is not None
            else None
        )

        context_list, mask_list = [], []
        for inference_guidance in self.guidance_at_inference:
            # All conditioning
            if inference_guidance == [1, 1]:
                context_list.append(context)
                if mask is not None:
                    mask_list.append(mask)
            # Zero-out the image conditioning
            elif inference_guidance == [0, 1]:
                context_list.append(torch.zeros_like(context))
                if mask is not None:
                    mask_list.append(mask)
            # Zero-out the latent conditioning
            elif inference_guidance == [1, 0]:
                context_list.append(context)
                if mask is not None:
                    mask_list.append(torch.zeros_like(mask))
            # Zero-out all conditioning
            elif inference_guidance == [0, 0]:
                context_list.append(torch.zeros_like(context))
                if mask is not None:
                    mask_list.append(torch.zeros_like(mask))
            else:
                raise Exception(f"Unknown guidance: {inference_guidance}")

        # Aggregate as a single batch
        context = torch.cat(context_list, dim=0)
        mask = torch.cat(mask_list, dim=0) if mask is not None else None

        return latent, context, mask, framestep

    def aggregate_cfg(self, aggregated: torch.Tensor) -> torch.Tensor:
        """
        Unroll the batch and aggregate the results with the cfg-scales.
        For 1-direction guidance:
            (1 - scale) * V[.|None] + scale * V[.|y]
        For 2-direction guidance:
            V[.|None] + scale1 * (V[.|y1,None] - V[.|None]) + scale2 * (V[.|y1,y2] - V[.|y1,None])
        """
        if not self.inference_enabled:
            return aggregated

        outputs_denoised = aggregated.chunk(len(self.guidance_at_inference), dim=0)
        output = outputs_denoised[0]

        assert len(outputs_denoised) == len(
            self.guidance_at_inference
        ), f"Invalid guidance: {len(outputs_denoised)} outputs, {len(self.guidance_at_inference)} guidance"

        for i in range(len(self.guidance_at_inference) - 1):
            output += self.guidance_scales[i] * (
                outputs_denoised[i + 1] - outputs_denoised[i]
            )

        return output
