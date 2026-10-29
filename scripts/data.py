import argparse
import shutil
from pathlib import Path

import kagglehub

KAGGLE_DATASET = "ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "classification_split"


def download_dataset() -> Path:
    """Download the dataset from Kaggle (kagglehub caches it and skips if already present)."""
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as error:
        raise RuntimeError(
            f"Download from Kaggle failed: {error}\n"
            "If authentication is required, set KAGGLE_USERNAME / KAGGLE_KEY "
            "environment variables or place kaggle.json in ~/.kaggle/."
        ) from error
    return Path(path)


def find_dataset_root(root: Path) -> Path:
    """Many datasets wrap everything in one top-level folder: unwrap it."""
    entries = [p for p in root.iterdir() if p.name != "__MACOSX"]
    while len(entries) == 1 and entries[0].is_dir():
        root = entries[0]
        entries = list(root.iterdir())
    return root


def main(force: bool = False) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if any(p for p in DATASET_DIR.iterdir() if p.name != ".gitkeep") and not force:
        print(
            "Dataset already present in data/classification_split/, "
            "skipping. Use --force to redo."
        )
        return

    downloaded_path = download_dataset()
    dataset_root = find_dataset_root(downloaded_path)

    for item in dataset_root.iterdir():
        if item.name == "__MACOSX":
            continue
        target = DATASET_DIR / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        # Copy (not move) so the kagglehub cache stays intact
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    print(f"Dataset ready in {DATASET_DIR}")
    print("Contents:", sorted(p.name for p in DATASET_DIR.iterdir()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the Kaggle dataset for classification training."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing data/classification_split dataset.",
    )
    main(force=parser.parse_args().force)
