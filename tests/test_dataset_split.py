from pathlib import Path

from PIL import Image

from src.final_project.dataset_split import create_dataset_split, stratified_split


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (48, 40), color=color).save(path)


def test_stratified_split_is_deterministic_and_has_no_overlap(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for breed in ("Abyssinian", "american_bulldog"):
        for number in range(1, 11):
            make_image(images_dir / f"{breed}_{number}.jpg", (20, 40, 60))

    images = sorted(images_dir.glob("*.jpg"))
    train_a, test_a = stratified_split(images, seed=42)
    train_b, test_b = stratified_split(images, seed=42)

    assert train_a == train_b
    assert test_a == test_b
    assert set(train_a).isdisjoint(test_a)
    assert set(train_a) | set(test_a) == set(images)
    assert len(test_a) == 4

    train_breeds = {path.stem.rpartition("_")[0] for path in train_a}
    test_breeds = {path.stem.rpartition("_")[0] for path in test_a}
    assert train_breeds == test_breeds == {"Abyssinian", "american_bulldog"}


def test_create_dataset_split_preserves_originals_and_writes_manifest(tmp_path):
    images_dir = tmp_path / "dataset" / "images"
    split_dir = tmp_path / "dataset" / "split"
    images_dir.mkdir(parents=True)
    for breed in ("Abyssinian", "american_bulldog"):
        for number in range(1, 6):
            make_image(images_dir / f"{breed}_{number}.jpg", (100, 120, 140))

    original_names = sorted(path.name for path in images_dir.iterdir())
    train_count, test_count = create_dataset_split(images_dir, split_dir, seed=42)

    assert train_count + test_count == len(original_names)
    assert len(list((split_dir / "train").glob("*.jpg"))) == train_count
    assert len(list((split_dir / "test").glob("*.jpg"))) == test_count
    assert sorted(path.name for path in images_dir.iterdir()) == original_names
    assert (split_dir / "manifest.json").is_file()
    assert (split_dir / "train.txt").is_file()
    assert (split_dir / "test.txt").is_file()


def test_split_rebuild_removes_stale_generated_files(tmp_path):
    images_dir = tmp_path / "images"
    split_dir = tmp_path / "split"
    images_dir.mkdir()
    for breed in ("Abyssinian", "american_bulldog"):
        for number in range(1, 4):
            make_image(images_dir / f"{breed}_{number}.jpg", (10, 20, 30))

    create_dataset_split(images_dir, split_dir)
    (split_dir / "test" / "stale.jpg").write_bytes(b"stale")
    create_dataset_split(images_dir, split_dir)

    assert not (split_dir / "test" / "stale.jpg").exists()
