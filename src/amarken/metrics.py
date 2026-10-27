"""Small reusable defaults; model modules choose their own evaluation metrics."""


def classification_metrics(num_classes: int):
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        MulticlassAccuracy,
        MulticlassCalibrationError,
        MulticlassConfusionMatrix,
        MulticlassF1Score,
        MulticlassPrecision,
        MulticlassRecall,
    )

    return MetricCollection(
        {
            "accuracy": MulticlassAccuracy(num_classes, average="micro"),
            "precision_macro": MulticlassPrecision(num_classes, average="macro"),
            "recall_macro": MulticlassRecall(num_classes, average="macro"),
            "f1_macro": MulticlassF1Score(num_classes, average="macro"),
            "f1_weighted": MulticlassF1Score(num_classes, average="weighted"),
            "precision_per_class": MulticlassPrecision(num_classes, average=None),
            "recall_per_class": MulticlassRecall(num_classes, average=None),
            "f1_per_class": MulticlassF1Score(num_classes, average=None),
            "confusion_matrix": MulticlassConfusionMatrix(num_classes),
            "calibration_error": MulticlassCalibrationError(
                num_classes, n_bins=15, norm="l1"
            ),
        }
    )
