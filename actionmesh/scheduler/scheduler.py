# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Scheduler implementations for ActionMesh.

Provides flow matching schedulers for the denoising process.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from actionmesh.scheduler.guidance import ClassifierFreeGuidance
from tqdm import tqdm


@dataclass(eq=False)
class SchedulerFlow:
    """
    Flow Matching Scheduler.

    Args:
        num_inference_steps: Number of denoising steps during inference.
        num_train_timesteps: Total number of training timesteps (default: 1000).
        shift: Shift value for the timestep schedule (default: 3.0).
            Higher values concentrate more steps at higher noise levels.
        is_additive: If True, use additive flow (x + dt*v), else subtractive (x - dt*v).
    """

    num_inference_steps: int
    num_train_timesteps: int = 1000
    shift: float = 3.0
    is_additive: bool = False

    def get_schedule(self) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute timesteps and step distances on demand.

        Returns:
            Tuple of (timesteps, distances) tensors.
        """
        timesteps = self._compute_timesteps(
            num_inference_steps=self.num_inference_steps + 1,
            num_train_timesteps=self.num_train_timesteps,
            shift=self.shift,
        )
        distances = (timesteps[:-1] - timesteps[1:]) / self.num_train_timesteps
        return timesteps, distances

    @staticmethod
    def _compute_timesteps(
        num_inference_steps: int,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ) -> torch.FloatTensor:
        """
        Compute timesteps for flow matching.

        Args:
            num_inference_steps: Number of inference/denoising steps.
            num_train_timesteps: Total number of training timesteps.
            shift: Shift value for the timestep schedule (1.0 = no shift).

        Returns:
            FloatTensor of shape (num_inference_steps,) containing the timesteps.
        """
        # Compute sigma_min from the full shifted schedule
        full_sigmas = (
            np.linspace(1, num_train_timesteps, num_train_timesteps)
            / num_train_timesteps
        )
        full_sigmas = full_sigmas[::-1]  # [1.0, ..., 1/N]
        full_sigmas_shifted = shift * full_sigmas / (1 + (shift - 1) * full_sigmas)
        sigma_max = full_sigmas_shifted[0]  # 1.0
        sigma_min = full_sigmas_shifted[-1]  # smallest shifted sigma

        # Create linearly spaced timesteps between sigma_max*N and sigma_min*N
        timesteps = np.linspace(
            sigma_max * num_train_timesteps,
            sigma_min * num_train_timesteps,
            num_inference_steps,
        )

        # Convert to sigmas and apply shift transformation
        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # Convert back to timesteps
        timesteps = (sigmas * num_train_timesteps).astype(np.float32)
        return torch.from_numpy(timesteps)

    def get_noise(
        self,
        latent_shape: list[int],
        batch_size: int,
        n_timesteps: int,
        device: str,
        generator: Optional[torch.Generator] = None,
        corr_noise: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate noise tensor with optional temporal correlation.

        Args:
            latent_shape: Shape of a single latent [N_TOKENS, WIDTH].
            n_timesteps: Number of timesteps/frames.
            generator: Optional random generator for reproducibility.
            corr_noise: Correlation factor (0.0 = independent, 1.0 = same noise).

        Returns:
            Noise tensor of shape [BATCH, N_TIMESTEPS, N_TOKENS, WIDTH].
        """
        assert 0 <= corr_noise <= 1.0

        same_noise = torch.randn(
            [batch_size, 1] + list(latent_shape),
            generator=generator,
            device=device,
        ).repeat(1, n_timesteps, 1, 1)

        indpt_noise = torch.randn(
            [batch_size, n_timesteps] + list(latent_shape),
            generator=generator,
            device=device,
        )

        return (
            math.sqrt(corr_noise) * same_noise + math.sqrt(1 - corr_noise) * indpt_noise
        )

    def _flow_sample(
        self,
        diffusion_model: torch.nn.Module,
        cf_guidance: ClassifierFreeGuidance,
        init_latent: torch.Tensor,
        context: torch.FloatTensor,
        device: torch.device = "cuda:0",
        disable_prog: bool = True,
        mask: Optional[torch.Tensor] = None,
        framestep: Optional[torch.Tensor] = None,
    ):
        """
        Flow matching sampling loop.

        Args:
            diffusion_model: The denoising model.
            cf_guidance: Classifier-free guidance configuration.
            init_latent: Initial noisy latent of shape (B, N_TOKENS, N_DIMS).
            context: Conditioning context of shape (B|2B, N_TOKENS_COND, W).
            device: Device to run on.
            disable_prog: Whether to disable progress bar.
            mask: Optional mask of shape (B|2B,).
            framestep: Optional frame step tensor of shape (B|2B,).

        Yields:
            Tuple of (latents, timestep, None) at each step.
        """
        # Initialize latents
        latents = init_latent

        # Compute schedule and move to device
        timesteps, distances = self.get_schedule()
        timesteps = timesteps.to(device)
        distances = distances.to(device)

        # Get the unobserved mask (None if not in CAT3D setting)
        unobserved = cf_guidance.get_unobserved_mask(mask)

        # Will be populated on first forward call and reused in subsequent calls
        freqs_rot = None

        for i, t in enumerate(
            tqdm(
                timesteps[:-1],
                disable=disable_prog,
                desc="Temporal 3D Denoising (Stage I)",
                leave=True,
            )
        ):
            # Expand the latents if we are doing classifier free guidance
            (hidden_states_input, context_input, mask_input, framestep_input) = (
                cf_guidance.cfg_at_inference(
                    latent=latents,
                    context=context,
                    mask=mask,
                    framestep=framestep,
                )
            )

            diffusion_time = torch.tensor(
                [t], dtype=latents.dtype, device=device
            ).expand(hidden_states_input.shape[0])

            # Predict the noise residual
            output_pred, freqs_rot = diffusion_model.forward(
                hidden_states=hidden_states_input,
                context=context_input,
                framestep=framestep_input,
                mask=mask_input,
                diffusion_time=diffusion_time,
                freqs_rot=freqs_rot,
            )

            # Perform guidance
            output_pred = cf_guidance.aggregate_cfg(output_pred)

            # Flow step
            if self.is_additive:
                flow_step_t = latents + distances[i] * output_pred
            else:
                flow_step_t = latents - distances[i] * output_pred

            # Apply mask if present
            if unobserved is not None:
                assert unobserved.any(), "No unobserved frames found"
                latents[unobserved] = flow_step_t[unobserved]
            else:
                latents = flow_step_t

            yield latents, t

    @torch.no_grad()
    def denoise(
        self,
        diffusion_model: torch.nn.Module,
        cf_guidance: ClassifierFreeGuidance,
        init_latent: torch.Tensor,
        context: torch.FloatTensor,
        device: torch.device = "cuda:0",
        disable_prog: bool = True,
        mask: Optional[torch.Tensor] = None,
        framestep: Optional[torch.Tensor] = None,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Run the denoising loop.

        Args:
            diffusion_model: The denoising model.
            cf_guidance: Classifier-free guidance.
            init_latent: Initial noisy latent.
            context: Conditioning context.
            mask: Optional mask to indicate conditioning latents.
            framestep: Optional frame step information.
            step_callback: Optional callback called at each step with (step, total_steps).

        Returns:
            Denoised latents.
        """
        sample_loop = self._flow_sample(
            diffusion_model=diffusion_model,
            cf_guidance=cf_guidance,
            init_latent=init_latent,
            context=context,
            device=device,
            disable_prog=disable_prog,
            mask=mask,
            framestep=framestep,
        )
        total_steps = self.num_inference_steps
        for step_idx, (sample, t) in enumerate(sample_loop):
            latents = sample
            if step_callback is not None:
                step_callback(step_idx + 1, total_steps)
        return latents
