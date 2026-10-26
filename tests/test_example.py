import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.main import loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything


class CheckpointTests(unittest.TestCase):
    def test_storage(self):
        for package in ("amarken",):
            with (
                self.subTest(package=package),
                tempfile.TemporaryDirectory() as directory,
            ):
                module = importlib.import_module(f"{package}.logger")
                root = Path(directory) / "checkpoints"

                def save(state, path):
                    Path(path).write_text(repr(state))

                torch = Mock(save=save)
                with (
                    patch.object(module, "CHECKPOINT_DIR", root),
                    patch.dict(sys.modules, torch=torch),
                ):
                    logger = module.WandbLogger(
                        enabled=False, checkpoints={"backend": "local"}
                    )
                    result = logger.save_checkpoint({"step": 1}, "run/latest.pt")
                    self.assertEqual(result, root / "run/latest.pt")
                    self.assertEqual(result.read_text(), "{'step': 1}")
                    for invalid in ("../escape.pt", "/tmp/escape.pt", "."):
                        with self.assertRaises(ValueError):
                            logger.save_checkpoint({}, invalid)
                    worker = module.WandbLogger(
                        enabled=False, rank=1, checkpoints={"backend": "local"}
                    )
                    self.assertIsNone(worker.save_checkpoint({}, "worker.pt"))
                    self.assertFalse((root / "worker.pt").exists())
                    bucket = module.WandbLogger(
                        enabled=False,
                        checkpoints={
                            "backend": "s3",
                            "bucket": "test-bucket",
                            "prefix": "/run-1/",
                        },
                    )
                    client = Mock()
                    bucket._storage_client = client
                    self.assertEqual(
                        bucket.save_checkpoint({}, "step.pt"),
                        "s3://test-bucket/run-1/step.pt",
                    )
                    client.upload_file.assert_called_once_with(
                        str(root / "step.pt"), "test-bucket", "run-1/step.pt"
                    )
                    client.upload_file.side_effect = RuntimeError("upload failed")
                    with self.assertRaises(RuntimeError):
                        bucket.save_checkpoint({}, "retained.pt")
                    self.assertTrue((root / "retained.pt").exists())
                    with (
                        patch.object(
                            torch, "save", side_effect=RuntimeError("save failed")
                        ),
                        self.assertRaises(RuntimeError),
                    ):
                        logger.save_checkpoint({}, "run/latest.pt")
                    self.assertEqual(result.read_text(), "{'step': 1}")
                    with self.assertRaises(ValueError):
                        module.CheckpointConfig(backend="s3")
                    with self.assertRaises(ValueError):
                        module.CheckpointConfig(backend="unknown")


class TrainerTests(unittest.TestCase):
    def make_trainer(self, epochs):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
        )
        inputs = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
        targets = torch.arange(12) % 2
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=True)
        config = TrainingConfig(epochs=epochs, run_name="test")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return Trainer(
            model,
            loader,
            loss_fn,
            config,
            optimizer=optimizer,
            scheduler=scheduler,
            validation_loader=loader,
        )

    def test_resume_matches_uninterrupted_training(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
        ):
            seed_everything(42)
            uninterrupted = self.make_trainer(2)
            history = uninterrupted.fit()
            seed_everything(42)
            first = self.make_trainer(1)
            first.fit()
            resumed = self.make_trainer(2)
            resumed.resume(Path(directory) / "test/epoch-0001.pt")
            resumed_history = resumed.fit()
            self.assertEqual(history[1], resumed_history[0])
            self.assertEqual(uninterrupted.global_step, resumed.global_step)
            self.assertEqual(
                uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict()
            )
            for key, expected in uninterrupted.model.state_dict().items():
                self.assertTrue(
                    torch.equal(expected, resumed.model.state_dict()[key]), key
                )

    def test_invalid_config(self):
        for kwargs in (
            {"epochs": 0},
            {"precision": "float16"},
            {"run_name": "../escape"},
            {"checkpoints": {"backend": "wandb"}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingConfig(**kwargs)
