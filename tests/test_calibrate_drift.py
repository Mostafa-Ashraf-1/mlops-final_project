from pathlib import Path

import numpy as np
from PIL import Image

from final_project.calibrate_drift import calibrate_threshold


def save_image(path: Path, value: int) -> None:
    pixels = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def test_calibration_is_reproducible_and_returns_threshold(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    for index, value in enumerate((20, 40, 60, 80, 100, 120, 140, 160)):
        save_image(clean_dir / f"pet_{index}.jpg", value)

    first = calibrate_threshold(clean_dir, repeats=8, seed=17)
    second = calibrate_threshold(clean_dir, repeats=8, seed=17)

    assert first["suggested_threshold"] == second["suggested_threshold"]
    assert first["clean_comparison_max_scores"] == second["clean_comparison_max_scores"]
    assert len(first["clean_comparison_max_scores"]) == 8
    assert 0.0 <= first["suggested_threshold"] <= 1.0
    assert first["image_count"] == 8


def test_calibration_requires_enough_images(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    save_image(clean_dir / "pet_1.jpg", 100)
    save_image(clean_dir / "pet_2.jpg", 120)
    save_image(clean_dir / "pet_3.jpg", 140)

    try:
        calibrate_threshold(clean_dir, repeats=2)
    except ValueError as error:
        assert "At least 4 clean images" in str(error)
    else:
        raise AssertionError("Expected insufficient-image validation to fail")
