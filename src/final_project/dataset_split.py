"""Create a deterministic, stratified train/test split for Oxford-IIIT Pet.

The original `images/` directory is never modified. The held-out test set is
reserved for final evaluation and corruption/drift experiments; it must not be
used for training or validation.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Sequence

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_TEST_SIZE = 0.20
DEFAULT_SEED = 42


def breed_from_filename(path: Path) -> str:
    """Parse a breed from names such as american_bulldog_12.jpg."""
    stem = path.stem
    breed, separator, image_number = stem.rpartition("_")
    if not separator or not image_number.isdigit() or not breed:
        raise ValueError(f"Unexpected Oxford-IIIT Pet filename: {path.name}")
    return breed


def stratified_split(
    images: Sequence[Path],
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[Path], list[Path]]:
    """Split images by breed, keeping each breed represented in both sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    by_breed: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_breed[breed_from_filename(image)].append(image)

    rng = random.Random(seed)
    train: list[Path] = []
    test: list[Path] = []

    for breed in sorted(by_breed):
        breed_images = sorted(by_breed[breed], key=lambda p: p.name)
        if len(breed_images) < 2:
            raise ValueError(
                f"Breed {breed!r} has fewer than two images; "
                "cannot represent it in both train and test."
            )
        rng.shuffle(breed_images)
        n_test = max(1, round(len(breed_images) * test_size))
        n_test = min(n_test, len(breed_images) - 1)
        test.extend(breed_images[:n_test])
        train.extend(breed_images[n_test:])

    return sorted(train, key=lambda p: p.name), sorted(test, key=lambda p: p.name)


def build_manifest(
    train_images: Sequence[Path],
    test_images: Sequence[Path],
    *,
    dataset_root: Path,
    output_path: Path,
    seed: int,
    test_size: float,
) -> None:
    """Write a JSON manifest recording paths, labels, dimensions, and split."""
    from PIL import Image

    records = []
    for split_name, paths in (("trainval", train_images), ("test", test_images)):
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
            breed = breed_from_filename(path)
            species = "cat" if breed[0].isupper() else "dog"
            records.append(
                {
                    "image_id": path.stem,
                    "path": path.relative_to(dataset_root).as_posix(),
                    "breed": breed,
                    "species": species,
                    "class_index": None,
                    "split": split_name,
                    "corruption": None,
                    "severity": 0,
                    "width": width,
                    "height": height,
                    "format": image_format,
                }
            )

    records.sort(key=lambda item: item["image_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "test_size": test_size,
                "num_trainval": len(train_images),
                "num_test": len(test_images),
                "images": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def create_dataset_split(
    images_dir: Path,
    split_dir: Path,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[int, int]:
    """Create split/train and split/test from images/, preserving images/."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )
    if not images:
        raise ValueError(f"No supported images found in {images_dir}")

    train_images, test_images = stratified_split(
        images, test_size=test_size, seed=seed
    )

    # Rebuild only generated split outputs, preventing stale files from
    # previous runs from silently contaminating the dataset.
    if split_dir.exists():
        shutil.rmtree(split_dir)
    train_dir = split_dir / "train"
    test_dir = split_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    for path in train_images:
        shutil.copy2(path, train_dir / path.name)
    for path in test_images:
        shutil.copy2(path, test_dir / path.name)

    dataset_root = images_dir.parent
    manifest_path = split_dir / "manifest.json"
    build_manifest(
        train_images,
        test_images,
        dataset_root=dataset_root,
        output_path=manifest_path,
        seed=seed,
        test_size=test_size,
    )

    # Save exact file membership separately for easy auditing/reproduction.
    (split_dir / "train.txt").write_text(
        "".join(f"{path.name}\n" for path in train_images), encoding="utf-8"
    )
    (split_dir / "test.txt").write_text(
        "".join(f"{path.name}\n" for path in test_images), encoding="utf-8"
    )
    return len(train_images), len(test_images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("Data/oxford-iiit-pet/images"),
        help="Original image directory (left untouched).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/oxford-iiit-pet/split"),
        help="Generated split directory.",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    train_count, test_count = create_dataset_split(
        args.images_dir,
        args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
    )
    total = train_count + test_count
    print(f"Created stratified split (seed={args.seed}):")
    print(f"  Train + validation: {train_count} ({train_count / total:.1%})")
    print(f"  Held-out test:      {test_count} ({test_count / total:.1%})")
    print(f"  Manifest:           {args.output_dir / 'manifest.json'}")
    print("Original images/ directory was not modified.")


if __name__ == "__main__":
    main()
