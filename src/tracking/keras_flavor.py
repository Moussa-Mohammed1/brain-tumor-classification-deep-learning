"""MLflow PyFunc flavor for the InceptionV3 brain-tumor classifier.

The classifier is a ``tf_keras`` model that embeds a TensorFlow Hub
``KerasLayer``. Serialising it through MLflow's native TensorFlow flavor is
fragile because the Hub layer is a custom object, so the checkpoint is wrapped
in a small PyFunc model instead. The result is a registered MLflow model that
can be reloaded with::

    import mlflow.pyfunc
    model = mlflow.pyfunc.load_model("models:/brainscan-classification/1")
    model.predict(["scan.jpg"])

Preprocessing mirrors ``src/classification/dataset.py``: RGB, resized to
299x299, pixel values scaled to [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.pyfunc


class KerasPyfuncModel(mlflow.pyfunc.PythonModel):
    """Expose an InceptionV3 ``.keras`` checkpoint through the PyFunc API."""

    def __init__(
        self,
        class_names: list | None = None,
        image_size: tuple = (299, 299),
        rescale: bool = True,
    ):
        self.class_names = list(class_names or [])
        self.image_size = tuple(image_size)
        self.rescale = bool(rescale)
        self.model = None

    def load_context(self, context) -> None:
        import tensorflow_hub as hub
        import tf_keras as keras

        self.model = keras.models.load_model(
            context.artifacts["model"],
            custom_objects={"KerasLayer": hub.KerasLayer},
        )

        classes_path = context.artifacts.get("classes")
        if classes_path:
            with open(classes_path, "r", encoding="utf-8") as file:
                classes = json.load(file)
            if classes:
                self.class_names = list(classes)

    def predict(self, context, model_input: Any):
        import numpy as np
        import pandas as pd

        images = self._load_images(model_input)
        if len(images) == 0:
            return pd.DataFrame(
                columns=[
                    "predicted_class_id",
                    "predicted_class",
                    "confidence",
                ]
            )

        probabilities = self.model.predict(images, verbose=0)

        rows = []
        for probability in probabilities:
            class_id = int(np.argmax(probability))
            class_name = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else str(class_id)
            )
            row = {
                "predicted_class_id": class_id,
                "predicted_class": class_name,
                "confidence": float(probability[class_id]),
            }
            for index, name in enumerate(self.class_names):
                row[f"probability_{name}"] = float(probability[index])
            rows.append(row)

        return pd.DataFrame(rows)

    def _load_images(self, model_input: Any):
        import numpy as np

        if isinstance(model_input, (str, Path)):
            items = [model_input]
        elif hasattr(model_input, "iloc") and hasattr(model_input, "columns"):
            items = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, np.ndarray):
            if model_input.ndim == 4:
                batch = model_input.astype("float32")
                if self.rescale and np.issubdtype(model_input.dtype, np.integer):
                    batch = batch / 255.0
                return batch
            items = [model_input]
        elif isinstance(model_input, (list, tuple)):
            items = list(model_input)
        else:
            items = [model_input]

        images = []
        for item in items:
            array = np.asarray(self._to_array(item))
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            images.append(array)

        return np.stack(images).astype("float32")

    def _to_array(self, item: Any):
        import numpy as np
        from PIL import Image

        if isinstance(item, Image.Image):
            image = item.convert("RGB")
        elif isinstance(item, (str, Path)):
            image = Image.open(item).convert("RGB")
        elif isinstance(item, (bytes, bytearray)):
            from io import BytesIO

            image = Image.open(BytesIO(item)).convert("RGB")
        else:
            array = np.asarray(item)
            if array.ndim == 3 and array.shape[-1] == 3:
                if not np.issubdtype(array.dtype, np.integer):
                    # Float arrays are assumed to already be in [0, 1].
                    return array.astype("float32")
                image = Image.fromarray(array.astype("uint8"))
            else:
                image = Image.fromarray(array.astype("uint8"))

        image = image.resize(self.image_size, Image.BILINEAR)
        array = np.asarray(image, dtype="float32")
        return array / (255.0 if self.rescale else 1.0)
