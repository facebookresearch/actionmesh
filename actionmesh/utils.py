# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""General utility functions for the actionmesh package."""

import gc
import os

import torch
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def force_memory_cleanup():
    """Force aggressive memory cleanup: GC + CUDA sync + cache clear."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def download_if_missing(
    repo_id: str,
    local_dir: str,
) -> str:
    """Download from HuggingFace Hub only if local directory doesn't exist.

    Args:
        repo_id: The repository ID on HuggingFace Hub.
        local_dir: Local directory to download to.

    Returns:
        Path to the local directory.
    """
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return local_dir


def load_config(config_name: str, config_dir: str, updates: dict = {}):
    """Load a Hydra configuration from a directory.

    Args:
        config_name: Name of the config file (without .yaml extension).
        config_dir: Directory containing the config files.
        updates: Dictionary of key-value pairs to update in the config.

    Returns:
        The loaded and resolved OmegaConf configuration.
    """
    with initialize_config_dir(
        config_dir=config_dir,
        version_base="1.1",
        job_name="load_config",
    ):
        cfg = compose(
            config_name=config_name,
            return_hydra_config=False,
            overrides=[
                "hydra.output_subdir=null",
                "hydra.job.chdir=false",
                "hydra/job_logging=none",
                "hydra/hydra_logging=none",
            ],
        )
        for k, v in updates.items():
            OmegaConf.update(cfg, k, v)
        OmegaConf.resolve(cfg)
    return cfg
