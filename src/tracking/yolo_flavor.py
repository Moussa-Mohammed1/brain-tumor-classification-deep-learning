"""MLflow PyFunc flavor for Ultralytics YOLO checkpoints.

Logging the detector with this wrapper produces a real MLflow model (with the
``MLmodel`` descriptor) that can be registered, versioned and reloaded with
``mlflow.pyfunc.load_model(...)`` without knowing how YOLO is instantiated.

Example
-------
>>> import mlflow.pyfunc
>>> model = mlflow.pyfunc.load_model("models:/brainscan-detection/1")
>>> model.predict(["scan.jpg"])
[{"class_name": "Glioma", "confidence": 0.91, "x1": 12, "y1": 20, ...}]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.pyfunc


class YoloPyfuncModel(mlflow.pyfunc.PythonModel):
    """Expose an Ultralytics YOLO checkpoint through the PyFunc interface."""

    def __init__(self, conf: float = 0.25, iou: float = 0.45):
        self.conf = float(conf)
        self.iou = float(iou)
        self.model = None
        self.class_names: list = []

    def load_context(self, context) -> None:
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])

        classes_path = Path(context.artifacts["classes"])
        with open(classes_path, "r", encoding="utf-8") as file:
            self.class_names = json.load(file)

    def predict(self, context, model_input: Any):
        sources = self._to_sources(model_input)
        if not sources:
            return []

        results = self.model.predict(
            source=sources,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        return [self._format_result(result) for result in results]

    @staticmethod
    def _to_sources(model_input: Any):
        """Normalise supported inputs to a list of paths/arrays for YOLO."""
        if isinstance(model_input, (str, Path)):
            return [str(model_input)]

        if hasattr(model_input, "iloc") and hasattr(model_input, "columns"):
            # pandas DataFrame: first column holds the image paths.
            return model_input.iloc[:, 0].tolist()

        if hasattr(model_input, "tolist") and not isinstance(model_input, list):
            values = model_input.tolist()
            return values if isinstance(values, list) else [values]

        if isinstance(model_input, (list, tuple)):
            return list(model_input)

        return [model_input]

    def _format_result(self, result) -> dict:
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)

        detections = []
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": names.get(
                            class_id,
                            self.class_names[class_id]
                            if class_id < len(self.class_names)
                            else str(class_id),
                        ),
                        "confidence": float(box.conf.item()),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    }
                )

        return {
            "image": str(getattr(result, "path", "")),
            "detections": detections,
            "num_detections": len(detections),
        }
