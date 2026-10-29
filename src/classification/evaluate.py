import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
import tf_keras as keras 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.classification.dataset import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "classification" / "best_model.keras"
REPORT_DIR = ROOT_DIR / "reports" / "classification"

def get_predictions(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
    return np.asarray(y_true), np.asarray(y_pred)

def evaluate_split(model, split_name, class_names):
    dataset = load_dataset(split_name, shuffle=False)
    loss, accuracy = model.evaluate(dataset, verbose=1)

    y_true, y_pred = get_predictions(model, dataset)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    results = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(
            report["weighted avg"]["precision"]
        ),
        "recall_weighted": float(
            report["weighted avg"]["recall"]
        ),
        "f1_weighted": float(
            report["weighted avg"]["f1-score"]
        ),
        "classification_report": report,
    }
    return results, y_true, y_pred
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )
    model = keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={"KerasLayer": hub.KerasLayer},
    )

    class_names = sorted(
        path.name
        for path in (
            ROOT_DIR / "data" / "classification_split" / "Train"
        ).iterdir() if path.is_dir()
    )

    all_results = {}
    predictions = {}

    for split_name in ["Train", "Validation", "Test"]:
        results, y_true, y_pred = evaluate_split(
            model, split_name, class_names
        )

        all_results[split_name] = results
        predictions[split_name] = (y_true, y_pred)

        print(f"\n--- {split_name} results ---")
        print(f"Loss:      {results['loss']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Recall (macro):    {results['recall_macro']:.4f}")
        print(f"F1-score (macro):  {results['f1_macro']:.4f}")
    metrics_path = REPORT_DIR / "evaluation_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(all_results, file, indent=4)

    y_true, y_pred = predictions["Test"]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    display.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("BrainScan — Test Set Confusion Matrix")
    fig.tight_layout()
    matrix_path = REPORT_DIR / "confusion_matrix_test.png"
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)

    split_names = ["Train", "Validation", "Test"]
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    print("\n--- Performance comparison ---")
    for metric in metric_names:
        values = [
            all_results[split][metric] for split in split_names
        ]
        print(
            f"{metric:18s} "
            + " | ".join(
                f"{split}: {value:.4f}"
                for split, value in zip(split_names, values)
            )
        )

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {matrix_path}")


if __name__ == "__main__":
    main()
