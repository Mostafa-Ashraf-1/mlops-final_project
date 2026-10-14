"""Estimate a practical drift threshold from clean-versus-clean comparisons.

This calibration measures the natural variation observed when two random groups
are sampled from the same clean held-out test set. It does not guarantee a
production false-positive rate; validate the threshold on separate data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from final_project.drift_detection import (
    DEFAULT_DRIFT_THRESHOLD,
    compare_feature_distributions,
    image_feature_matrix,
    supported_images,
)

DEFAULT_REPEATS = 100
DEFAULT_SEED = 42
DEFAULT_PERCENTILE = 95.0


def calibrate_threshold(
    clean_dir: Path,
    *,
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
    percentile: float = DEFAULT_PERCENTILE,
) -> dict:
    """Split clean images randomly many times and summarize maximum KS scores."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be greater than 0 and at most 100.")

    images = supported_images(clean_dir)
    if len(images) < 4:
        raise ValueError(
            "At least 4 clean images are required to form two groups of 2 or more."
        )

    rng = random.Random(seed)
    scores = []
    feature_scores = []

    for repeat in range(repeats):
        shuffled = images.copy()
        rng.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        group_a = shuffled[:midpoint]
        group_b = shuffled[midpoint:]

        features_a = image_feature_matrix(group_a)
        features_b = image_feature_matrix(group_b)
        comparison = compare_feature_distributions(
            features_a, features_b, threshold=1.0
        )
        scores.append(comparison["max_drift_score"])
        feature_scores.append(comparison["feature_scores"])

    suggested_threshold = float(np.percentile(scores, percentile))
    return {
        "reference_directory": str(clean_dir),
        "image_count": len(images),
        "repeats": repeats,
        "seed": seed,
        "percentile": percentile,
        "method": "percentile of maximum per-feature KS distance from clean-vs-clean splits",
        "suggested_threshold": suggested_threshold,
        "existing_threshold": DEFAULT_DRIFT_THRESHOLD,
        "clean_comparison_max_scores": scores,
        "max_score_min": float(np.min(scores)),
        "max_score_median": float(np.median(scores)),
        "max_score_95th_percentile": float(np.percentile(scores, 95)),
        "max_score_max": float(np.max(scores)),
        "feature_scores_by_repeat": feature_scores,
        "warning": (
            "Exploratory calibration only. Reusing the same clean images in "
            "many splits does not provide independent production validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=Path("Data/oxford-iiit-pet/split/test"),
        help="Directory containing uncorrupted held-out test images.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/drift_baseline.json"),
    )
    args = parser.parse_args()

    report = calibrate_threshold(
        args.clean_dir,
        repeats=args.repeats,
        seed=args.seed,
        percentile=args.percentile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Clean images: {report['image_count']}")
    print(f"Comparisons: {report['repeats']}")
    print(f"Maximum KS scores: min={report['max_score_min']:.3f}, "
          f"median={report['max_score_median']:.3f}, "
          f"max={report['max_score_max']:.3f}")
    print(
        f"Suggested threshold (p{report['percentile']:.0f}): "
        f"{report['suggested_threshold']:.3f}"
    )
    print(f"Current threshold: {report['existing_threshold']:.3f}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
