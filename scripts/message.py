import shutil
from pathlib import Path

import kagglehub

from src.config import RAW_DATA_DIR, TMP_DIR

KAGGLE_DATASET = "ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes"


def download_dataset() -> Path:
    """Download the dataset from Kaggle (kagglehub caches it and skips if already present)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET)
    except Exception as e:
        raise RuntimeError(
            f"Download from Kaggle failed: {e}\n"
            "If authentication is required, set KAGGLE_USERNAME / KAGGLE_KEY "
            "environment variables or place kaggle.json in ~/.kaggle/."
        ) from e
    return Path(path)


def find_dataset_root(root: Path) -> Path:
    """Many datasets wrap everything in one top-level folder: unwrap it."""
    entries = [p for p in root.iterdir() if p.name != "__MACOSX"]
    while len(entries) == 1 and entries[0].is_dir():
        root = entries[0]
        entries = list(root.iterdir())
    return root


def main(force: bool = False) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if any(p for p in RAW_DATA_DIR.iterdir() if p.name != ".gitkeep") and not force:
        print("Data already present in data/raw/, skipping. Use --force to redo.")
        return

    downloaded_path = download_dataset()
    dataset_root = find_dataset_root(downloaded_path)

    for item in dataset_root.iterdir():
        target = RAW_DATA_DIR / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        # Copy (not move) so the kagglehub cache stays intact
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"Dataset ready in {RAW_DATA_DIR}")
    print("Contents:", sorted(p.name for p in RAW_DATA_DIR.iterdir()))


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)