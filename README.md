<div align="center">


<h1>🎬 ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion</h1>

<a href="https://remysabathier.github.io/actionmesh/actionmesh_2026.pdf" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Paper-ActionMesh" alt="Paper PDF"></a>
<a href="https://arxiv.org/abs/2601.16148"><img src="https://img.shields.io/badge/arXiv-2601.16148-b31b1b" alt="arXiv"></a>
<a href="https://remysabathier.github.io/actionmesh/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href='https://huggingface.co/spaces/facebook/ActionMesh'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href="https://colab.research.google.com/github/facebookresearch/ActionMesh/blob/main/notebooks/ActionMesh.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

**[Meta Reality Labs](https://ai.facebook.com/research/)**;  **[SpAItial](https://www.spaitial.ai/)**; **[University College London](https://geometry.cs.ucl.ac.uk/)**

[Remy Sabathier](https://remysabathier.github.io/RemySabathier/), [David Novotny](https://d-novotny.github.io/), [Niloy J. Mitra](https://geometry.cs.ucl.ac.uk/), [Tom Monnier](https://tmonnier.com/)
</div>

<img src="assets/docs/teaser.jpg" alt="ActionMesh teaser" width="100%">

## 📖 Overview

**ActionMesh** is a **fast Video** → **Animated 3D Mesh** model that generates an animated 3D mesh (topology fixed) from input videos (real or synthetic).


## 🆕 Updates
- **2026-01-31**: 🆕 Low RAM mode (`--low_ram`) — ActionMesh can now runs on Google Colab T4 GPUs! [Try it on Colab](https://colab.research.google.com/github/facebookresearch/ActionMesh/blob/main/notebooks/ActionMesh.ipynb)

- **2025-01-21**: Demo is live! Try it here: [🤗 facebook/ActionMesh](https://huggingface.co/spaces/facebook/ActionMesh)

- **2025-01-21**: Code released!



## ⚙️ Installation

### Requirements

- **GPU**: NVIDIA GPU with 32GB VRAM (tested on A100, H100, and H200)
- **GPU (Low RAM)**: 🆕 Supports GPUs with 12GB VRAM using `--low_ram` mode (e.g., Google Colab T4)
- **PyTorch**: Requires PyTorch and torchvision (developed with torch 2.4.0 / CUDA 12.1 and torchvision 0.19.0)

### 1. Clone and Install Dependencies

```bash
git clone git@github.com:facebookresearch/actionmesh.git
cd actionmesh
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
```

### 2. Optional Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| **PyTorch3D** | Video rendering of animated meshes | [Installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) |
| **Blender 3.5.1** | Export animated mesh as a single `.glb` file | [Download](https://download.blender.org/release/Blender3.5/) |


## 🚀 Quick Start

### Basic Usage

Generate an animated mesh from an input video:
> Note: To export a single animated mesh file (importable in Blender), specify the path to your Blender executable via --blender_path.



```bash
python inference/video_to_animated_mesh.py \
    --input assets/examples/davis_camel \
    --blender_path "path/to/blender/executable"  # optional: export animated mesh for Blender
```

### Fast & Low RAM Modes

For faster inference (as used in the [HuggingFace demo](https://huggingface.co/spaces/facebook/ActionMesh)):

```bash
python inference/video_to_animated_mesh.py \
    --input assets/examples/davis_camel \
    --fast \
    --blender_path "path/to/blender/executable"
```

For low RAM GPUs (e.g., Google Colab T4):

```bash
python inference/video_to_animated_mesh.py \
    --input assets/examples/davis_camel \
    --fast --low_ram \
    --blender_path "path/to/blender/executable"
```

**Performance comparison on H100 GPU:**

| Mode | Time | Quality |
|------|------|---------|
| Default | ~115s | Higher quality |
| Fast (`--fast`) | ~45s | Slightly reduced quality |

### Model Downloads

On the first launch, ActionMesh weights and external models are automatically downloaded from HuggingFace:

| Model | Source | Local Path |
|-------|--------|------------|
| **ActionMesh** | [facebook/ActionMesh](https://huggingface.co/facebook/ActionMesh) | `pretrained_weights/ActionMesh` |
| **TripoSG** (image-to-3D) | [VAST-AI/TripoSG](https://huggingface.co/VAST-AI/TripoSG) | `pretrained_weights/TripoSG` |
| **DinoV2** | [facebook/dinov2-large](https://huggingface.co/facebook/dinov2-large) | `pretrained_weights/dinov2` |
| **RMBG** | [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | `pretrained_weights/RMBG` |


## 🎨 Examples

We provide example sequences in `assets/examples/` with expected outputs for testing and debugging your installation:

| Example | Type | Expected Output |
|---------|------|-----------------|
| `davis_camel` | ![](https://img.shields.io/badge/Real-2ea44f) [![](https://img.shields.io/badge/DAVIS-blue)](https://davischallenge.org/) | <img src="assets/docs/camel_renders.gif" width="300"> |
| `davis_flamingo` | ![](https://img.shields.io/badge/Real-2ea44f) [![](https://img.shields.io/badge/DAVIS-blue)](https://davischallenge.org/) | <img src="assets/docs/flamingo_renders.gif" width="300"> |
| `kangaroo` | ![](https://img.shields.io/badge/Synthetic-a855f7) | <img src="assets/docs/kangaroo_renders.gif" width="300"> |
| `spring` | ![](https://img.shields.io/badge/Synthetic-a855f7) | <img src="assets/docs/spring_renders.gif" width="300"> |


## 🎬 Input

The `--input` argument accepts:
- A `.mp4` video file
- A folder containing PNG images

The number of input frames should be between **16** and **31** (default is 16). Any additional frames will be ignored.

### Masks

Input frames can be provided with or without alpha masks. If no mask is provided, [RMBG](https://huggingface.co/briaai/RMBG-1.4) background removal model is automatically applied to each frame before processing.

> **Tip:** For custom videos, we strongly recommend using the [SAM2 demo](https://sam2.metademolab.com/demo) to isolate the animated subject on a white background, as RMBG may have limited performance on complex scenes. See our [SAM2 extraction guide](assets/docs/sam2_extraction_guide.md) for detailed instructions.


## 📦 Export

The model exports a folder containing:

| Output | Description | Requirements |
|--------|-------------|--------------|
| **Per-frame meshes** | One `.glb` mesh file per timestep (`mesh_000.glb`, `mesh_001.glb`, ...) | None (default) |
| **Animated mesh** | Single `animated_mesh.glb` with embedded animation, importable in Blender | [Blender 3.5.1](https://download.blender.org/release/Blender3.5/) |
| **Video** | Rendered `.mp4` video of the animated mesh | [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) |

<details>
<summary><b>🎥 Video output preview</b></summary>
<br>
<img src="assets/docs/camel_renders.gif" alt="Video output example" width="480">
</details>

<details open>
<summary><b>🎞️ Animated mesh file imported in Blender</b></summary>
<br>
<img src="assets/docs/blender_export.gif" alt="Blender export example" width="480">
</details>



## 🏛️ License

See the LICENSE file for details about the license under which this code is made available.


## 🙏 Acknowledgements

ActionMesh builds upon the following open-source projects. We thank the authors for making their work available:

| Project | Description |
|---------|-------------|
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) | Image-to-3D mesh generation |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Self-supervised vision features |
| [Diffusers](https://github.com/huggingface/diffusers) | Diffusion model framework |
| [Transformers](https://github.com/huggingface/transformers) | Transformer model library |
| [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | Background removal model |

## 📚 Citation

```
@inproceedings{ActionMesh2025,
author = {Remy Sabathier, David Novotny, Niloy Mitra, Tom Monnier},
title = {ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion},
year = {2025},
}
```
