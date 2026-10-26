import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.main import loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything


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
            resumed.resume(Path(directory) / first.checkpoint_prefix / "latest.pt")
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

    def test_checkpoint_policies(self):
        self.assertEqual(TrainingConfig().checkpoints.policy, "default")
        for policy, filenames in (
            ("best", {"best.pt"}),
            ("default", {"best.pt", "latest.pt"}),
            ("all", {"epoch-0002.pt", "epoch-0003.pt"}),
        ):
            with (
                self.subTest(policy=policy),
                tempfile.TemporaryDirectory() as directory,
                patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
            ):
                trainer = self.make_trainer(3)
                trainer.scheduler = None
                trainer.config.checkpoints = type(trainer.config.checkpoints)(
                    backend="local", policy=policy, every_n_epochs=2
                )
                values = iter([3.0, 3.0, 2.0, 1.0, 1.0, 2.0])
                with patch.object(
                    trainer,
                    "_epoch_loss",
                    side_effect=values,
                ):
                    trainer.fit()
                root = Path(directory) / trainer.checkpoint_prefix
                self.assertEqual({p.name for p in root.iterdir()}, filenames)
                if policy != "all":
                    self.assertEqual(
                        torch.load(root / "best.pt", weights_only=True)["epoch"], 2
                    )
                if policy == "default":
                    self.assertEqual(
                        torch.load(root / "latest.pt", weights_only=True)["epoch"], 3
                    )
