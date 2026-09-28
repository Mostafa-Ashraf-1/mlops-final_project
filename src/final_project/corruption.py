
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split


BASE_PATH = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_PATH = BASE_PATH / "artifacts"
DATA_PATH = BASE_PATH / "Data" / "oxford-iiit-pet"
IMAGES_PATH = DATA_PATH  / "images"
SPLIT_DIR = DATA_PATH / "split"
TRAIN_DIR = SPLIT_DIR / "train"
TEST_DIR = SPLIT_DIR / "test"
CORRUPTION_DIR = DATA_PATH / "corruption_test"
# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# Dataset utilities
# ============================================================

def get_class_name(path: Path) -> str:
    """
    Extract the breed name from an Oxford-IIIT Pet filename.

    Examples:

        Abyssinian_1.jpg
            -> Abyssinian

        american_bulldog_12.jpg
            -> american_bulldog

        staffordshire_bull_terrier_7.jpg
            -> staffordshire_bull_terrier
    """

    return path.stem.rsplit("_", 1)[0]


def get_images():
    """Load all images from the original dataset."""

    images = [
        path
        for path in IMAGES_PATH.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise RuntimeError(
            f"No images found in {IMAGES_PATH}"
        )

    labels = [
        get_class_name(path)
        for path in images
    ]

    return images, labels


# ============================================================
# Train / Test split
# ============================================================

def create_split(images, labels):
    """
    Create a stratified 80/20 train/test split.

    The test set is completely held out from training.
    """

    train_images, test_images = train_test_split(
        images,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return train_images, test_images


def copy_split(
    train_images,
    test_images,
):
    """
    Copy the images into:

        split/train/
        split/test/

    The original images/ directory is never modified.
    """

    TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCreating train split...")

    for image_path in train_images:

        destination = TRAIN_DIR / image_path.name

        shutil.copy2(
            image_path,
            destination,
        )

    print("\nCreating test split...")

    for image_path in test_images:

        destination = TEST_DIR / image_path.name

        shutil.copy2(
            image_path,
            destination,
        )


# ============================================================
# Corruptions
# ============================================================

def gaussian_blur(
    image: Image.Image,
    severity: int,
) -> Image.Image:
    """
    Simulates an out-of-focus phone camera.

    Severity:
        1 -> sigma 1
        2 -> sigma 2
        3 -> sigma 4
    """

    sigma = {
        1: 1.0,
        2: 2.0,
        3: 4.0,
    }[severity]

    array = np.array(image)

    kernel_size = int(
        2 * round(3 * sigma) + 1
    )

    blurred = cv2.GaussianBlur(
        array,
        (kernel_size, kernel_size),
        sigmaX=sigma,
    )

    return Image.fromarray(blurred)


def brightness_shift(
    image: Image.Image,
    severity: int,
    direction: str,
) -> Image.Image:
    """
    Simulates lighting changes.

    Up:
        1 -> +10%
        2 -> +25%
        3 -> +40%

    Down:
        1 -> -10%
        2 -> -25%
        3 -> -40%
    """

    amount = {
        1: 0.10,
        2: 0.25,
        3: 0.40,
    }[severity]

    if direction == "up":
        factor = 1.0 + amount

    elif direction == "down":
        factor = 1.0 - amount

    else:
        raise ValueError(
            "direction must be 'up' or 'down'"
        )

    enhancer = ImageEnhance.Brightness(image)

    return enhancer.enhance(factor)


def jpeg_compression(
    image: Image.Image,
    severity: int,
) -> Image.Image:
    """
    Simulates messaging-app JPEG re-encoding.

    Severity:
        1 -> quality 70
        2 -> quality 50
        3 -> quality 30
    """

    from io import BytesIO

    quality = {
        1: 70,
        2: 50,
        3: 30,
    }[severity]

    image = image.convert("RGB")

    buffer = BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def downscale_upscale(
    image: Image.Image,
    severity: int,
) -> Image.Image:
    """
    Simulates an old / low-quality handset.

    Severity:
        1 -> 192x192
        2 -> 128x128
        3 -> 96x96

    The image is restored to its original dimensions
    afterward.
    """

    target_size = {
        1: 192,
        2: 128,
        3: 96,
    }[severity]

    original_size = image.size

    downscaled = image.resize(
        (target_size, target_size),
        Image.Resampling.BILINEAR,
    )

    restored = downscaled.resize(
        original_size,
        Image.Resampling.BILINEAR,
    )

    return restored


def motion_blur(
    image: Image.Image,
    severity: int,
) -> Image.Image:
    """
    Simulates camera movement.

    Severity:
        1 -> kernel 5
        2 -> kernel 9
        3 -> kernel 15
    """

    kernel_size = {
        1: 5,
        2: 9,
        3: 15,
    }[severity]

    array = np.array(image)

    kernel = np.zeros(
        (kernel_size, kernel_size)
    )

    kernel[kernel_size // 2, :] = (
        1.0 / kernel_size
    )

    blurred = cv2.filter2D(
        array,
        -1,
        kernel,
    )

    return Image.fromarray(blurred)


# ============================================================
# Corruption dispatcher
# ============================================================

def apply_corruption(
    image: Image.Image,
    corruption: str,
    severity: int,
) -> Image.Image:

    if corruption == "gaussian_blur":
        return gaussian_blur(
            image,
            severity,
        )

    if corruption == "brightness_up":
        return brightness_shift(
            image,
            severity,
            "up",
        )

    if corruption == "brightness_down":
        return brightness_shift(
            image,
            severity,
            "down",
        )

    if corruption == "jpeg":
        return jpeg_compression(
            image,
            severity,
        )

    if corruption == "downscale":
        return downscale_upscale(
            image,
            severity,
        )

    if corruption == "motion_blur":
        return motion_blur(
            image,
            severity,
        )

    raise ValueError(
        f"Unknown corruption: {corruption}"
    )


CORRUPTIONS = [
    "gaussian_blur",
    "brightness_up",
    "brightness_down",
    "jpeg",
    "downscale",
    "motion_blur",
]


# ============================================================
# Generate corruption suite
# ============================================================

def generate_corruption_suite(
    test_images,
):
    """
    Generate:

        clean
        corruption/severity_1
        corruption/severity_2
        corruption/severity_3

    using ONLY the held-out test set.
    """

    # --------------------------------------------------------
    # Clean test set
    # --------------------------------------------------------

    clean_dir = CORRUPTION_DIR / "clean"

    clean_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCreating clean test set...")

    for image_path in test_images:

        shutil.copy2(
            image_path,
            clean_dir / image_path.name,
        )

    # --------------------------------------------------------
    # Corruptions
    # --------------------------------------------------------

    for corruption in CORRUPTIONS:

        print(
            f"\nGenerating {corruption}..."
        )

        for severity in range(1, 4):

            print(
                f"  Severity {severity}"
            )

            output_dir = (
                CORRUPTION_DIR
                / corruption
                / f"severity_{severity}"
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for image_path in test_images:

                image = Image.open(
                    image_path
                ).convert("RGB")

                corrupted = apply_corruption(
                    image,
                    corruption,
                    severity,
                )

                output_path = (
                    output_dir
                    / image_path.name
                )

                corrupted.save(
                    output_path,
                    quality=95,
                )


# ============================================================
# Ground-truth manifest
# ============================================================

def create_manifest(test_images):
    """
    Create a CSV containing the ground truth for
    every corruption scenario.
    """

    import csv

    CORRUPTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        CORRUPTION_DIR / "manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "filename",
                "class",
                "corruption",
                "severity",
            ]
        )

        for image_path in test_images:

            class_name = get_class_name(
                image_path
            )

            # Clean
            writer.writerow(
                [
                    image_path.name,
                    class_name,
                    "clean",
                    0,
                ]
            )

            # Corrupted versions
            for corruption in CORRUPTIONS:

                for severity in range(1, 4):

                    writer.writerow(
                        [
                            image_path.name,
                            class_name,
                            corruption,
                            severity,
                        ]
                    )

    print(
        f"\nManifest saved to: {manifest_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Oxford-IIIT Pet Corruption Suite")
    print("=" * 60)

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not IMAGES_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{IMAGES_PATH}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        f"\nLoading images from:\n{IMAGES_PATH}"
    )

    images, labels = get_images()

    print(
        f"Total images: {len(images)}"
    )

    print(
        f"Total classes: {len(set(labels))}"
    )

    # --------------------------------------------------------
    # Create stratified 80/20 split
    # --------------------------------------------------------

    train_images, test_images = create_split(
        images,
        labels,
    )

    print(
        f"\nTrain images: {len(train_images)} "
        f"({len(train_images) / len(images) * 100:.2f}%)"
    )

    print(
        f"Test images: {len(test_images)} "
        f"({len(test_images) / len(images) * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Copy train/test split
    # --------------------------------------------------------

    copy_split(
        train_images,
        test_images,
    )

    # --------------------------------------------------------
    # Generate corruptions ONLY from test
    # --------------------------------------------------------

    generate_corruption_suite(
        test_images
    )

    # --------------------------------------------------------
    # Generate ground truth
    # --------------------------------------------------------

    create_manifest(
        test_images
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print("\nDataset split:")
    print(
        f"  Train: {TRAIN_DIR}"
    )
    print(
        f"  Test:  {TEST_DIR}"
    )

    print(
        "\nCorruption suite:"
    )
    print(
        f"  {CORRUPTION_DIR}"
    )

    print(
        "\nOriginal dataset was NOT modified."
    )


if __name__ == "__main__":
    main()
