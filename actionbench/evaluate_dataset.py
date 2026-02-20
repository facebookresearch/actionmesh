# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for computing 3D/4D metrics across a dataset.

Usage:
    python evaluate_dataset.py \
        --gt_root /path/to/gt/root \
        --pred_root /path/to/pred/root \
        --output_csv /path/to/results.csv \
        --device cuda

Expected structure:
    GT:   {gt_root}/{uid}/surfaces.npy     (T, N, 6) point clouds
    Pred: {pred_root}/{uid}/mesh_*.glb    or {pred_root}/{uid}/*.glb

Note:
    When using HuggingFace ActionBench, gt_root should point to the
    data/ subdirectory (e.g. data/actionbench/data/).
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from benchmark import compute_chamfer_3d_4d
from natsort import natsorted
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SampleResult:
    """Result for a single sample evaluation."""

    uid: str
    cd_3d: float = float("nan")
    cd_4d: float = float("nan")
    cd_motion: float = float("nan")
    n_frames: int = 0
    status: str = "pending"
    error_message: str = ""


@dataclass
class DatasetResults:
    """Aggregated results for the entire dataset."""

    samples: list[SampleResult] = field(default_factory=list)

    def add(self, result: SampleResult) -> None:
        self.samples.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(s) for s in self.samples])

    def summary(self) -> dict[str, float]:
        """Compute mean metrics over successful samples."""
        df = self.to_dataframe()
        success_df = df[df["status"] == "success"]

        n_total = len(df)
        n_success = len(success_df)
        n_failed = n_total - n_success

        summary = {
            "n_total": n_total,
            "n_success": n_success,
            "n_failed": n_failed,
            "success_rate": n_success / n_total if n_total > 0 else 0.0,
        }

        if n_success > 0:
            summary["cd_3d_mean"] = success_df["cd_3d"].mean()
            summary["cd_4d_mean"] = success_df["cd_4d"].mean()
            summary["cd_motion_mean"] = success_df["cd_motion"].mean()
        else:
            summary["cd_3d_mean"] = float("nan")
            summary["cd_4d_mean"] = float("nan")
            summary["cd_motion_mean"] = float("nan")

        return summary


def find_uids(gt_root: Path, pred_root: Path) -> list[str]:
    """
    Find all UIDs that exist in both GT and prediction directories.

    Args:
        gt_root: Root directory containing GT surfaces.
        pred_root: Root directory containing predictions.

    Returns:
        List of UIDs found in both directories.
    """
    # Find GT UIDs (directories with surfaces.npy)
    gt_uids = {p.parent.name for p in gt_root.glob("*/surfaces.npy")}

    # Find pred UIDs (directories with mesh files)
    pred_uids = {p.parent.name for p in pred_root.glob("*/*.glb")}
    pred_uids |= {p.parent.name for p in pred_root.glob("*/*.obj")}

    common_uids = gt_uids & pred_uids

    logger.info(
        f"Found {len(gt_uids)} GT, {len(pred_uids)} pred, " f"{len(common_uids)} common"
    )

    if gt_uids - pred_uids:
        logger.warning(f"Missing predictions: {len(gt_uids - pred_uids)}")

    if pred_uids - gt_uids:
        logger.warning(f"Missing GT: {len(pred_uids - gt_uids)}")

    return sorted(common_uids)


def load_gt_surfaces(gt_path: Path) -> torch.Tensor:
    """
    Load GT surfaces from a .npy file.

    Args:
        gt_path: Path to surfaces.npy file.

    Returns:
        (T, N, 3) tensor of point cloud positions.
    """
    data = torch.from_numpy(np.load(gt_path))
    return data[..., :3].float()


def load_pred_meshes(
    pred_dir: Path,
    n_frames: int | None = None,
    pattern: str = "mesh_*.glb",
) -> list[trimesh.Trimesh]:
    """
    Load predicted meshes from a directory.

    Args:
        pred_dir: Directory containing mesh files.
        n_frames: Expected number of frames. If None, loads all found.
        pattern: Glob pattern to match mesh files.

    Returns:
        List of trimesh meshes ordered by filename.
    """
    mesh_files = natsorted(pred_dir.glob(pattern))

    if not mesh_files:
        raise FileNotFoundError(f"No mesh files found in {pred_dir}")

    if n_frames is not None:
        if len(mesh_files) < n_frames:
            raise ValueError(
                f"Not enough meshes: found {len(mesh_files)}, need {n_frames}"
            )
        mesh_files = mesh_files[:n_frames]

    return [trimesh.load(p, force="mesh") for p in mesh_files]


def evaluate_sample(
    uid: str,
    gt_root: Path,
    pred_root: Path,
    device: str = "cuda",
    n_pts_icp: int = 10_000,
    n_pts_chamfer: int = 100_000,
    seed: int = 44,
    mesh_pattern: str = "mesh_*.glb",
) -> SampleResult:
    """
    Evaluate a single sample.

    Args:
        uid: Unique identifier for the sample.
        gt_root: Root directory containing GT surfaces.
        pred_root: Root directory containing predictions.
        device: Device for computation.
        n_pts_icp: Number of points for ICP alignment.
        n_pts_chamfer: Number of points for Chamfer computation.
        seed: Random seed.
        mesh_pattern: Glob pattern to match mesh files.

    Returns:
        SampleResult with computed metrics or error info.
    """
    result = SampleResult(uid=uid)

    try:
        gt_path = gt_root / uid / "surfaces.npy"
        pred_dir = pred_root / uid

        if not gt_path.exists():
            result.status = "error"
            result.error_message = f"GT not found: {gt_path}"
            return result

        if not pred_dir.exists():
            result.status = "error"
            result.error_message = f"Pred dir not found: {pred_dir}"
            return result

        gt_pc = load_gt_surfaces(gt_path)
        n_frames = gt_pc.shape[0]
        result.n_frames = n_frames

        try:
            pred_meshes = load_pred_meshes(
                pred_dir, n_frames=n_frames, pattern=mesh_pattern
            )
        except (FileNotFoundError, ValueError) as e:
            result.status = "error"
            result.error_message = str(e)
            return result

        cd_3d, cd_4d, cd_motion = compute_chamfer_3d_4d(
            gt_pc=gt_pc,
            pred_meshes=pred_meshes,
            device=device,
            is_4D=True,
            n_pts_icp=n_pts_icp,
            n_pts_chamfer=n_pts_chamfer,
            seed=seed,
        )

        result.cd_3d = cd_3d
        result.cd_4d = cd_4d
        result.cd_motion = cd_motion
        result.status = "success"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
        logger.error(f"[{uid}] Error: {e}")

    return result


def load_existing_results(output_csv: Path) -> dict[str, SampleResult]:
    """
    Load existing results from CSV for resumption.

    Args:
        output_csv: Path to the results CSV file.

    Returns:
        Dictionary mapping uid to SampleResult.
    """
    if not output_csv.exists():
        return {}

    df = pd.read_csv(output_csv)
    results = {}
    for _, row in df.iterrows():
        results[row["uid"]] = SampleResult(
            uid=row["uid"],
            cd_3d=row["cd_3d"],
            cd_4d=row["cd_4d"],
            cd_motion=row["cd_motion"],
            n_frames=row["n_frames"],
            status=row["status"],
            error_message=row.get("error_message", ""),
        )
    return results


def save_results(results: DatasetResults, output_path: Path) -> None:
    """
    Save results to CSV and summary JSON.

    Args:
        results: DatasetResults to save.
        output_path: Path to save the CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = results.to_dataframe()
    df.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(results.summary(), f, indent=2)


def evaluate_dataset(
    gt_root: str,
    pred_root: str,
    output_csv: str | None = None,
    device: str = "cuda",
    n_pts_icp: int = 10_000,
    n_pts_chamfer: int = 100_000,
    seed: int = 44,
    mesh_pattern: str = "mesh_*.glb",
    recompute: bool = False,
) -> DatasetResults:
    """
    Evaluate all samples in the dataset.

    Supports stop/start resilience: results are saved after each sample,
    and existing results are loaded on restart (unless recompute=True).

    Args:
        gt_root: Root directory containing GT surfaces.
        pred_root: Root directory containing predictions.
        output_csv: Optional path to save results CSV.
        device: Device for computation.
        n_pts_icp: Number of points for ICP alignment.
        n_pts_chamfer: Number of points for Chamfer computation.
        seed: Random seed.
        mesh_pattern: Glob pattern to match mesh files.
        recompute: If True, recompute all samples even if already done.

    Returns:
        DatasetResults containing all sample results and summary.
    """
    gt_root = Path(gt_root)
    pred_root = Path(pred_root)
    output_path = Path(output_csv) if output_csv else None

    uids = find_uids(gt_root, pred_root)

    # Load existing results for resumption
    existing_results: dict[str, SampleResult] = {}
    if output_path and not recompute:
        existing_results = load_existing_results(output_path)
        if existing_results:
            n_done = sum(1 for r in existing_results.values() if r.status == "success")
            logger.info(
                f"Loaded {len(existing_results)} existing results "
                f"({n_done} successful). Use --recompute to redo all."
            )

    results = DatasetResults()

    for uid in tqdm(uids, desc="Evaluating samples"):
        # Skip if already successfully computed
        if uid in existing_results and not recompute:
            prev = existing_results[uid]
            if prev.status == "success":
                results.add(prev)
                continue
            logger.info(f"[{uid}] Retrying previously failed sample")

        result = evaluate_sample(
            uid=uid,
            gt_root=gt_root,
            pred_root=pred_root,
            device=device,
            n_pts_icp=n_pts_icp,
            n_pts_chamfer=n_pts_chamfer,
            seed=seed,
            mesh_pattern=mesh_pattern,
        )
        results.add(result)

        if result.status == "success":
            logger.info(
                f"[{uid}] CD_3D={result.cd_3d:.3f}, "
                f"CD_4D={result.cd_4d:.3f}, "
                f"CD_Motion={result.cd_motion:.3f}"
            )

        # Save after each sample for resilience
        if output_path:
            save_results(results, output_path)

    if output_path:
        save_results(results, output_path)
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"Summary saved to: {output_path.with_suffix('.summary.json')}")

    return results


def print_summary(results: DatasetResults) -> None:
    """Print a formatted summary of the evaluation results."""
    summary = results.summary()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\nSamples:")
    print(f"  Total:   {summary['n_total']}")
    print(f"  Success: {summary['n_success']}")
    print(f"  Failed:  {summary['n_failed']}")
    print(f"  Rate:    {summary['success_rate']:.1%}")

    if summary["n_success"] > 0:
        print("\nMetrics (mean):")
        print(f"  CD_3D:     {summary['cd_3d_mean']:.3f}")
        print(f"  CD_4D:     {summary['cd_4d_mean']:.3f}")
        print(f"  CD_Motion: {summary['cd_motion_mean']:.3f}")

    # Report failed samples
    df = results.to_dataframe()
    failed = df[df["status"] != "success"]
    if len(failed) > 0:
        print(f"\nFailed samples ({len(failed)}):")
        for _, row in failed.iterrows():
            print(f"  [{row['uid']}] {row['status']}: {row['error_message']}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D/4D reconstruction metrics across a dataset"
    )
    parser.add_argument(
        "--gt_root",
        type=str,
        required=True,
        help="Root directory containing GT surfaces (uid/surfaces.npy)",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        required=True,
        help="Root directory containing predictions (uid/mesh_*.glb)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save results CSV",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation (cuda or cpu)",
    )
    parser.add_argument(
        "--n_pts_icp",
        type=int,
        default=10_000,
        help="Number of points for ICP alignment",
    )
    parser.add_argument(
        "--n_pts_chamfer",
        type=int,
        default=100_000,
        help="Number of points for Chamfer computation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--mesh_pattern",
        type=str,
        default="mesh_*.glb",
        help="Glob pattern for mesh files (default: mesh_*.glb)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all samples even if already done",
    )

    args = parser.parse_args()

    results = evaluate_dataset(
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        output_csv=args.output_csv,
        device=args.device,
        n_pts_icp=args.n_pts_icp,
        n_pts_chamfer=args.n_pts_chamfer,
        seed=args.seed,
        mesh_pattern=args.mesh_pattern,
        recompute=args.recompute,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
