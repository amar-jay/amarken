import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.logger import CheckpointConfig, WandbLogger
from amarken.main import loss_fn
from amarken.trainer import Trainer, TrainingConfig, seed_everything
from amarken.utils import prompt_run_name


class TrainerTests(unittest.TestCase):
    def make_trainer(self, epochs):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
        )
        inputs = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
        targets = torch.arange(12) % 2
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=True)
        config = TrainingConfig(
            epochs=epochs,
            run_name="test",
            checkpoints=CheckpointConfig(backend="local", async_save=True),
        )
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
        self.assertTrue(CheckpointConfig().overwrite)
        with self.assertRaises(ValueError):
            CheckpointConfig(policy="all")
        with self.assertRaises(ValueError):
            CheckpointConfig(overwrite="false")
        config = TrainingConfig(checkpoints={"backend": "local", "overwrite": False})
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("amarken.utils.questionary.text") as prompt,
        ):
            prompt.return_value.ask.return_value = "my-run"
            self.assertEqual(prompt_run_name(config), "my-run")
            validator = prompt.call_args.kwargs["validate"]
            self.assertIs(validator("my-run"), True)
            self.assertIsInstance(validator("../escape"), str)
            prompt.return_value.ask.return_value = "  "
            self.assertIsNone(prompt_run_name(config))
            prompt.return_value.ask.return_value = None
            with self.assertRaises(SystemExit) as cancelled:
                prompt_run_name(config)
            self.assertEqual(cancelled.exception.code, 130)
            prompt.reset_mock()
            self.assertIsNone(prompt_run_name(config, resume=True))
            self.assertIsNone(prompt_run_name(TrainingConfig()))
            with patch("sys.stdin.isatty", return_value=False):
                self.assertIsNone(prompt_run_name(config))
            prompt.assert_not_called()

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
        ):
            trainer = self.make_trainer(1)
            trainer.config.checkpoints = CheckpointConfig(
                backend="local", overwrite=False
            )
            with (
                patch(
                    "amarken.utils.prompt_run_name", return_value="fit-run"
                ) as prompt,
                patch.object(trainer, "_fit", return_value=[]) as training,
            ):
                prompt.assert_not_called()
                self.assertFalse((Path(directory) / "test/fit-run").exists())
                trainer.fit()
                prompt.assert_called_once_with(trainer.config, resume=False)
                self.assertEqual(trainer.run_label, "fit-run")
                self.assertTrue((Path(directory) / "test/fit-run").is_dir())
                trainer.fit()
                self.assertEqual(prompt.call_count, 1)
                self.assertEqual(training.call_count, 2)

        for policy in ("best", "default"):
            for overwrite in (True, False):
                with (
                    self.subTest(policy=policy, overwrite=overwrite),
                    tempfile.TemporaryDirectory() as directory,
                    patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
                ):
                    filenames = (
                        {"best.pt"} if policy == "best" else {"best.pt", "latest.pt"}
                    )
                    trainer = self.make_trainer(3)
                    trainer.scheduler = None
                    trainer.run_label = "named-run" if not overwrite else None
                    trainer.config.checkpoints = CheckpointConfig(
                        backend="local",
                        policy=policy,
                        overwrite=overwrite,
                        async_save=True,
                    )
                    with patch.object(
                        trainer,
                        "_epoch_loss",
                        side_effect=[3.0, 3.0, 2.0, 1.0, 1.0, 2.0],
                    ):
                        trainer.fit()
                    root = Path(directory) / trainer.checkpoint_prefix
                    self.assertEqual({p.name for p in root.iterdir()}, filenames)
                    self.assertEqual(
                        torch.load(root / "best.pt", weights_only=True)["epoch"], 2
                    )
                    if policy == "default":
                        self.assertEqual(
                            torch.load(root / "latest.pt", weights_only=True)["epoch"],
                            3,
                        )
                    repeated = self.make_trainer(1)
                    repeated.scheduler = None
                    repeated.config.checkpoints = trainer.config.checkpoints
                    if overwrite:
                        self.assertEqual(
                            repeated.checkpoint_prefix, trainer.checkpoint_prefix
                        )
                        with patch.object(
                            repeated, "_epoch_loss", side_effect=[4.0, 4.0]
                        ):
                            repeated.fit()
                        self.assertEqual(
                            {
                                p.relative_to(Path(directory)).as_posix()
                                for p in Path(directory).rglob("*.pt")
                            },
                            {f"test/{name}" for name in filenames},
                        )
                    else:
                        self.assertEqual(trainer.checkpoint_prefix, "test/named-run")
                        self.assertEqual(
                            repeated.checkpoint_prefix, f"test/{repeated.run_id}"
                        )
                        repeated.run_label = trainer.run_label
                        with self.assertRaises(FileExistsError):
                            repeated.fit()
                        repeated.resume(root / "best.pt")
                        self.assertEqual(repeated.run_label, "named-run")
                        self.assertEqual(
                            repeated.checkpoint_prefix, trainer.checkpoint_prefix
                        )
                        repeated.config.epochs = 3
                        with patch.object(
                            repeated, "_epoch_loss", side_effect=[4.0, 4.0]
                        ):
                            repeated.fit()

        # A held writer proves snapshot isolation while training state mutates.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
        ):
            entered, release = Event(), Event()
            original_save = torch.save

            def slow_save(state, path):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test writer was not released")
                original_save(state, path)

            logger = WandbLogger(
                enabled=False,
                checkpoints=CheckpointConfig(
                    backend="local", async_save=True, max_pending=2
                ),
            )
            state = {"weights": torch.tensor([1.0]), "nested": [{"step": 1}]}
            try:
                with patch("torch.save", side_effect=slow_save):
                    first = logger.save_checkpoint(state, "snapshot.pt")
                    self.assertTrue(entered.wait(5))
                    self.assertFalse(first.done())
                    state["weights"].add_(10)
                    state["nested"][0]["step"] = 2
                    second = logger.save_checkpoint(state, "second.pt")
                    self.assertLessEqual(len(logger._pending_checkpoints), 2)
                    release.set()
                    logger.finish()
                self.assertTrue(first.done() and second.done())
                snapshot = torch.load(first.result(), weights_only=True)
                self.assertEqual(snapshot["weights"].item(), 1)
                self.assertEqual(snapshot["nested"][0]["step"], 1)
                self.assertEqual(
                    torch.load(second.result(), weights_only=True)["weights"].item(), 11
                )
                with self.assertRaises(RuntimeError):
                    logger.save_checkpoint({}, "closed.pt")
            finally:
                release.set()
                logger.finish()

        # With one slot, the second call waits before allocating its snapshot.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
        ):
            logger = WandbLogger(
                enabled=False,
                checkpoints=CheckpointConfig(
                    backend="local", async_save=True, max_pending=1
                ),
            )
            entered, release = Event(), Event()
            try:
                with patch("torch.save", side_effect=slow_save):
                    first = logger.save_checkpoint({"step": 1}, "first.pt")
                    self.assertTrue(entered.wait(5))
                    original_result = first.result

                    def unblock_on_backpressure(*args, **kwargs):
                        release.set()
                        return original_result(*args, **kwargs)

                    with patch.object(
                        first, "result", side_effect=unblock_on_backpressure
                    ) as waiting:
                        second = logger.save_checkpoint({"step": 2}, "second.pt")
                        self.assertGreaterEqual(waiting.call_count, 1)
                    logger.finish()
                    self.assertTrue(second.done())
            finally:
                release.set()
                logger.finish()

        # Upload failures are observable, with local state retained and worker closed.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
        ):
            logger = WandbLogger(
                enabled=False,
                checkpoints=CheckpointConfig(
                    backend="s3", bucket="test-bucket", async_save=True
                ),
            )
            logger._storage_client = Mock()
            logger._storage_client.upload_file.side_effect = OSError("upload failed")
            future = logger.save_checkpoint({"step": 1}, "retained.pt")
            with self.assertRaisesRegex(OSError, "upload failed"):
                future.result()
            with self.assertRaisesRegex(OSError, "upload failed"):
                logger.check_checkpoint_errors()
            with self.assertRaisesRegex(OSError, "upload failed"):
                logger.finish()
            self.assertTrue((Path(directory) / "retained.pt").exists())
            self.assertIsNone(logger._checkpoint_executor)

        # W&B waits inside the ordered worker before a reused file is replaced.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
        ):
            logger = WandbLogger(
                enabled=False,
                checkpoints=CheckpointConfig(backend="wandb", async_save=True),
            )
            logger.enabled = True
            logger.run = Mock()
            entered, release = Event(), Event()
            artifact = Mock()

            def wait_upload():
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test upload was not released")

            artifact.wait.side_effect = wait_upload
            try:
                with patch.object(logger, "log_artifact", return_value=artifact):
                    first = logger.save_checkpoint({"step": 1}, "latest.pt")
                    self.assertTrue(entered.wait(5))
                    second = logger.save_checkpoint({"step": 2}, "latest.pt")
                    self.assertEqual(
                        torch.load(Path(directory) / "latest.pt", weights_only=True)[
                            "step"
                        ],
                        1,
                    )
                    release.set()
                    logger.finish()
                    self.assertTrue(first.done() and second.done())
                    self.assertEqual(artifact.wait.call_count, 2)
                    logger.run.finish.assert_called_once_with(exit_code=0)
            finally:
                release.set()
                logger.finish()
