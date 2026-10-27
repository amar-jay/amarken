"""Dataset settings stored in the training JSON and checkpoint metadata."""

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    id: str
    train_size: int = 128
    val_size: int = 32
    test_size: int = 32
    config: str | None = None
    test_fraction: float = 0.5
    train_split: str = "train"
    validation_split: str = "validation"
    text_column: str = "sentence"
    label_column: str = "label"

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("dataset.id must be a nonempty string")
        if self.config is not None and (
            not isinstance(self.config, str) or not self.config.strip()
        ):
            raise ValueError("dataset.config must be a nonempty string or null")
        for name in ("train_split", "validation_split", "text_column", "label_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"dataset.{name} must be a nonempty string")
        for name in ("train_size", "val_size", "test_size"):
            size = getattr(self, name)
            if type(size) is not int or size < 1:
                raise ValueError(f"dataset.{name} must be a positive integer")
        if (
            type(self.test_fraction) not in {int, float}
            or not 0 < self.test_fraction < 1
        ):
            raise ValueError(
                "dataset.test_fraction must be between 0 and 1 (exclusive)"
            )
