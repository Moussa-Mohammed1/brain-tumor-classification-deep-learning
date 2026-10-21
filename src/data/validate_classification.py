from pathlib import Path
import pandas as pd 
from PIL import Image, UnidentifiedImageError

DATASET_DIR = Path("data/classification")
ALLOWED_EXTENSIONS = {".jpeg", ".jpg", ".bmp", ".png"}

def load_dataset(dataset_dir: Path) -> pd.DataFrame:
    records = []
    
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found {dataset_dir}"
        )

    for file_path in dataset_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            print(f"[REJECTED: extension] {file_path}")
            continue
        if file_path.parent.name.lower() != "images":
            continue
        try:
            with Image.open(file_path) as image:
                image.verify()

            records.append({
                "image_path": str(file_path),
                "filename": file_path.name,
                "extension": file_path.suffix.lower(),
                "class_name": file_path.parent.parent.name,
            })

        except (OSError, ValueError, UnidentifiedImageError) as err:
            print(f"[REJECTED: invalid image] {file_path} - {err}")

    return pd.DataFrame(
        records,
        columns=["image_path", "filename", "extension", "class_name"],
    )

if __name__ == "__main__":
    dataset = load_dataset(DATASET_DIR)

    print(f"\nValid images: {len(dataset)}")

    if not dataset.empty:
        print("\nImages per class:")
        print(dataset["class_name"].value_counts())

        print("\nImage extensions:")
        print(dataset["extension"].value_counts())