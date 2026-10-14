"""Detect image-distribution drift using lightweight image statistics.

The clean test images are the reference distribution. Each corruption/severity
directory is compared with that reference using the two-sample Kolmogorov-Smirnov
statistic computed feature by feature. This is an input-drift signal, not a
measure of model accuracy degradation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_DRIFT_THRESHOLD = 0.10


def extract_image_features(image_path: Path) -> dict[str, float]:
    """Extract brightness, contrast, sharpness, and RGB channel statistics."""
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    gray = (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    )
    # Variance of the discrete Laplacian is a lightweight sharpness proxy.
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )

    features = {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(laplacian.var()),
    }
    for index, channel in enumerate(("red", "green", "blue")):
        values = rgb[..., index]
        features[f"{channel}_mean"] = float(values.mean())
        features[f"{channel}_std"] = float(values.std())
    return features


def image_feature_matrix(image_paths: Iterable[Path]) -> dict[str, np.ndarray]:
    """Return one numeric feature array per feature for a set of images."""
    records = [extract_image_features(path) for path in sorted(image_paths)]
    if not records:
        raise ValueError("No supported images found in the supplied directory.")
    return {
        feature_name: np.asarray(
            [record[feature_name] for record in records], dtype=np.float64
        )
        for feature_name in records[0]
    }


def ks_statistic(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov distance (0 to 1)."""
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    candidate = np.sort(np.asarray(candidate, dtype=np.float64))
    if reference.size == 0 or candidate.size == 0:
        raise ValueError("Both samples must contain at least one value.")

    values = np.sort(np.concatenate((reference, candidate)))
    ref_cdf = np.searchsorted(reference, values, side="right") / reference.size
    cand_cdf = np.searchsorted(candidate, values, side="right") / candidate.size
    return float(np.max(np.abs(ref_cdf - cand_cdf)))


def compare_feature_distributions(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    """Compare distributions and flag drift when any feature exceeds threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    scores = {
        feature: ks_statistic(reference[feature], candidate[feature])
        for feature in sorted(reference.keys() & candidate.keys())
    }
    if not scores:
        raise ValueError("Reference and candidate have no common features.")

    return {
        "drift_detected": any(score > threshold for score in scores.values()),
        "threshold": threshold,
        "max_drift_score": max(scores.values()),
        "feature_scores": scores,
    }


def supported_images(directory: Path) -> list[Path]:
    """List supported image files directly inside a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_drift_report(
    corruption_dir: Path,
    *,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
) -> dict:
    """Compare every corruption/severity folder against corruption_test/clean."""
    clean_dir = corruption_dir.parent / "split" / "test"
    reference = image_feature_matrix(supported_images(clean_dir))
    results = {}

    for scenario_dir in sorted(corruption_dir.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name == "clean":
            continue
        for candidate_dir in sorted(scenario_dir.iterdir()):
            if not candidate_dir.is_dir() or not candidate_dir.name.startswith("severity_"):
                continue
            candidate_images = supported_images(candidate_dir)
            if not candidate_images:
                raise ValueError(f"No images found in {candidate_dir}")
            candidate = image_feature_matrix(candidate_images)
            key = f"{scenario_dir.name}/{candidate_dir.name}"
            results[key] = {
                "image_count": len(candidate_images),
                **compare_feature_distributions(
                    reference, candidate, threshold=threshold
                ),
            }

    if not results:
        raise ValueError(f"No corruption severity directories found in {corruption_dir}")

    return {
        "reference": str(clean_dir),
        "reference_image_count": len(supported_images(clean_dir)),
        "method": "per-feature two-sample KS distance",
        "threshold": threshold,
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corruption-dir",
        type=Path,
        default=Path("Data/oxford-iiit-pet/corruption_test"),
        help="Directory containing clean/ and corruption/severity_N/ folders.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help="Flag drift when any feature's KS distance exceeds this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/drift_report.json"),
        help="Path for the JSON report.",
    )
    args = parser.parse_args()

    report = run_drift_report(args.corruption_dir, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Compared {len(report['scenarios'])} corruption scenarios.")
    print(f"Report saved to: {args.output}")
    for scenario, result in report["scenarios"].items():
        status = "DRIFT" if result["drift_detected"] else "no drift"
        print(f"{scenario}: {status} (max KS={result['max_drift_score']:.3f})")


if __name__ == "__main__":
    main()
