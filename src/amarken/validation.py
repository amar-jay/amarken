"""Evaluate a checkpoint on the selected dataset's validation split."""

from pathlib import Path

import torch

from amarken.dataset import build_dataloaders, build_dataset, build_test_dataset
from amarken.model import build_metrics, load_model, validation_step
from amarken.trainer import TrainingConfig, seed_everything


def run_validation(
    checkpoint: str | Path, config: TrainingConfig, *, split: str = "val"
) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    seed_everything(config.seed)
    model = load_model(checkpoint, device=config.device, precision=config.precision)
    model.eval()
    if split == "test":
        validation = build_test_dataset(
            seed=config.seed,
            dataset_id=config.dataset_id,
            dataset_config=config.dataset_config,
            test_size=config.test_size,
        )
        train = validation  # Only the evaluation loader is iterated.
    else:
        train, validation = build_dataset(
            seed=config.seed,
            dataset_id=config.dataset_id,
            dataset_config=config.dataset_config,
            train_size=config.train_size,
            val_size=config.val_size,
        )
    _, loader = build_dataloaders(
        train,
        validation,
        batch_size=config.batch_size,
        tokenize=getattr(model, "tokenize", None),
    )
    totals = {}
    samples = 0
    device = torch.device(config.device)
    task_metrics = build_metrics().to(device)
    with torch.inference_mode():
        for batch in loader:
            count = len(batch[1])
            with torch.autocast(
                device_type=device.type,
                dtype=getattr(torch, config.precision),
                enabled=config.precision != "float32",
            ):
                metrics = validation_step(model, batch, device, task_metrics)
            for name, value in metrics.items():
                value = float(value)
                if not torch.isfinite(torch.tensor(value)).item():
                    raise ValueError(f"Validation metric {name} must be finite")
                totals[name] = totals.get(name, 0.0) + value * count
            samples += count
    if not samples:
        raise ValueError("Validation split must contain at least one sample")
    result = {
        "samples": samples,
        **{f"{split}/{k}": v / samples for k, v in totals.items()},
    }
    for name, value in task_metrics.compute().items():
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Validation metric {name} must be finite")
        result[f"{split}/{name}"] = value.detach().cpu().tolist()
    return result
