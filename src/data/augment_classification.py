from pathlib import Path
import random
import shutil

import cv2
import numpy as np


DATASET_DIR = Path("data/classification_split")
TRAIN_DIR = DATASET_DIR / "Train"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def apply_augmentation(image, technique):
    """Applique une technique d'augmentation à une image."""

    if technique == "blur":
        return cv2.GaussianBlur(image, (3, 3), 0)

    if technique == "noise":
        noise = np.random.normal(
            loc=0,
            scale=10,
            size=image.shape
        )
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    if technique == "flip":
        
        return cv2.flip(image, 1)

    raise ValueError(f"Technique inconnue : {technique}")


def get_images(class_dir):
    images_dir = class_dir / "images"

    if not images_dir.is_dir():
        return []

    return [
        path for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
        and not path.stem.endswith(("_blur", "_noise", "_flip"))
    ]


def augment_dataset():
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Dossier d'entraînement introuvable : {TRAIN_DIR.resolve()}"
        )

    classes = {
        folder.name: get_images(folder)
        for folder in TRAIN_DIR.iterdir()
        if folder.is_dir()
    }

    classes = {
        name: images for name, images in classes.items()
        if images
    }

    if not classes:
        raise ValueError("Aucune image d'entraînement trouvée.")

  
    target_count = max(len(images) for images in classes.values())

    print("Nombre initial d'images :")
    for name, images in classes.items():
        print(f"  {name}: {len(images)}")

    print(f"\nObjectif par classe : {target_count}")

    for class_name, images in classes.items():
        class_dir = TRAIN_DIR / class_name
        images_dir = class_dir / "images"

        current_count = len(images)
        needed = target_count - current_count

        if needed <= 0:
            print(f"{class_name}: déjà équilibrée")
            continue

        print(f"{class_name}: génération de {needed} images")

        for index in range(needed):
            source_path = random.choice(images)

            image = cv2.imread(str(source_path))
            if image is None:
                print(f"Image illisible, ignorée : {source_path}")
                continue

            technique = random.choice(["blur", "noise", "flip"])
            augmented = apply_augmentation(image, technique)

            output_path = images_dir / (
                f"{source_path.stem}_{technique}_{index}"
                f"{source_path.suffix.lower()}"
            )

       
            if output_path.exists():
                continue

            if not cv2.imwrite(str(output_path), augmented):
                print(f"Échec d'écriture : {output_path}")

    print("\nAugmentation terminée.")
    print("Vérifie maintenant le nombre d'images par classe.")


if __name__ == "__main__":
    augment_dataset()