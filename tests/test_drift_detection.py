from pathlib import Path

import numpy as np
from PIL import Image

from final_project.drift_detection import (
    compare_feature_distributions,
    extract_image_features,
    image_feature_matrix,
    ks_statistic,
    run_drift_report,
)


def save_rgb(path: Path, value: int) -> None:
    pixels = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_extract_image_features_returns_expected_statistics(tmp_path: Path) -> None:
    image_path = tmp_path / "gray.jpg"
    save_rgb(image_path, 128)

    features = extract_image_features(image_path)

    assert set(features) == {
        "brightness",
        "contrast",
        "sharpness",
        "red_mean",
        "red_std",
        "green_mean",
        "green_std",
        "blue_mean",
        "blue_std",
    }
    assert 0.0 <= features["brightness"] <= 1.0
    assert features["contrast"] >= 0.0
    assert features["sharpness"] >= 0.0


def test_ks_statistic_is_zero_for_identical_samples() -> None:
    sample = np.array([0.1, 0.2, 0.3, 0.4])

    assert ks_statistic(sample, sample) == 0.0


def test_distribution_comparison_detects_shift() -> None:
    reference = {"brightness": np.array([0.1, 0.1, 0.2, 0.2])}
    candidate = {"brightness": np.array([0.8, 0.8, 0.9, 0.9])}

    result = compare_feature_distributions(
        reference, candidate, threshold=0.5
    )

    assert result["drift_detected"] is True
    assert result["feature_scores"]["brightness"] == 1.0


def test_run_drift_report_compares_each_severity_to_clean(
    tmp_path: Path,
) -> None:
    corruption_dir = tmp_path / "corruption_test"
    clean_dir = corruption_dir / "clean"
    blur_dir = corruption_dir / "gaussian_blur" / "severity_1"
    clean_dir.mkdir(parents=True)
    blur_dir.mkdir(parents=True)

    save_rgb(clean_dir / "pet_1.jpg", 30)
    save_rgb(clean_dir / "pet_2.jpg", 35)
    save_rgb(blur_dir / "pet_1.jpg", 220)
    save_rgb(blur_dir / "pet_2.jpg", 225)

    report = run_drift_report(corruption_dir, threshold=0.5)

    assert report["reference_image_count"] == 2
    assert "gaussian_blur/severity_1" in report["scenarios"]
    assert report["scenarios"]["gaussian_blur/severity_1"]["drift_detected"] is True
