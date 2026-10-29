from pathlib import Path

from PIL import Image

from src.data.validate_classification import load_dataset


def test_load_dataset_reads_all_images_and_their_class_names(tmp_path: Path) -> None:
    for class_name in ("Glioma", "Pituitary"):
        image_dir = tmp_path / "Train" / class_name / "images"
        image_dir.mkdir(parents=True)
        Image.new("RGB", (1, 1)).save(image_dir / f"{class_name.lower()}.jpg")

    dataset = load_dataset(tmp_path)

    assert len(dataset) == 2
    assert set(dataset["class_name"]) == {"Glioma", "Pituitary"}