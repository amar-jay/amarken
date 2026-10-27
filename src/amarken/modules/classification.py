"""Optional text classifier derived from base; replace labels/head for your task."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor, nn

from amarken.constants import CLASSIFICATION_FREEZE_BACKBONE, CLASSIFICATION_LABELS
from amarken.modules.base import BaseConfig, BaseModel
from amarken.modules.base import build_model as build_backbone
from amarken.modules.base import fsdp_modules as backbone_fsdp_modules
from amarken.utils import load_checkpoint_weights


class TextClassifier(nn.Module):
    def __init__(self, backbone: BaseModel):
        super().__init__()
        if len(CLASSIFICATION_LABELS) < 2:
            raise ValueError("CLASSIFICATION_LABELS must contain at least two classes")
        self.backbone = backbone
        self.freeze_backbone = CLASSIFICATION_FREEZE_BACKBONE
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.zero_grad(set_to_none=True)
        parameter = next(backbone.parameters())
        self.head = nn.Linear(backbone.hidden_size, len(CLASSIFICATION_LABELS)).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def tokenize(self, texts: Sequence[str]) -> dict[str, Tensor]:
        return self.backbone.tokenize(texts)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        features = self.backbone(inputs)
        # Direct predict() calls also work outside the trainer's autocast context.
        return self.head(features.to(dtype=self.head.weight.dtype))


def build_model(*, device: str = "cpu", precision: str = "float32") -> TextClassifier:
    return TextClassifier(build_backbone(device=device, precision=precision))


def load_model(
    checkpoint: str | Path, *, device: str = "cpu", precision: str = "float32"
) -> TextClassifier:
    weights = load_checkpoint_weights(checkpoint)
    if "head.weight" not in weights:
        raise ValueError(
            "Expected a classification checkpoint with trained head weights"
        )
    backbone = BaseModel.load(BaseConfig(dtype=precision), pretrained=False)
    model = TextClassifier(backbone)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def predict(model: TextClassifier, inputs: str | Sequence[str]) -> dict:
    texts = [inputs] if isinstance(inputs, str) else inputs
    batch = model.tokenize(texts)
    model.eval()
    with torch.inference_mode():
        probabilities = model(batch).float().softmax(-1)
    predictions = probabilities.argmax(-1).cpu().tolist()
    return {
        "predictions": predictions,
        "labels": [CLASSIFICATION_LABELS[index] for index in predictions],
        "probabilities": probabilities.cpu().tolist(),
    }


def loss_fn(model: TextClassifier, batch, device):
    """Cross-entropy on raw class logits; batch is (tokenized_inputs, class_ids)."""
    inputs, targets = batch
    if not isinstance(inputs, Mapping):
        raise TypeError("Classification inputs must be a tokenized dictionary")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return torch.nn.functional.cross_entropy(model(inputs).float(), targets.to(device))


def build_metrics():
    """Fresh dataset-level metrics; replace these for another training objective."""
    from amarken.metrics import classification_metrics

    return classification_metrics(len(CLASSIFICATION_LABELS))


def validation_step(model, batch, device, metrics) -> dict:
    """Update task metrics and return a batch-mean loss from one forward pass."""
    inputs, targets = batch
    inputs = {key: value.to(device) for key, value in inputs.items()}
    targets = targets.to(device)
    logits = model(inputs).float()
    metrics.update(logits.softmax(dim=-1), targets)
    return {"loss": torch.nn.functional.cross_entropy(logits, targets).item()}


def fsdp_modules(model):
    return [*backbone_fsdp_modules(model.backbone), model.head, model]
