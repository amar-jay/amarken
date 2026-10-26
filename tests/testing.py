import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.cli import main as cli_main
from amarken.inference import run_inference
from amarken.logger import CheckpointConfig, WandbLogger
from amarken.modules.example import build_model, load_model, loss_fn
from amarken.modules.example import predict as predict_numeric
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
            patch("amarken.inference.load_model", load_model),
            patch("amarken.inference.predict", predict_numeric),
        ):
            inference_config = Path(directory) / "inference-config.json"
            inference_config.write_text("{}")

            def infer_cli(arguments):
                if "--config" not in arguments:
                    arguments = [*arguments, "--config", str(inference_config)]
                cli_main(arguments)

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

            # Load actual resumed weights without constructing an optimizer/logger.
            with patch(
                "amarken.modules.example.build_model",
                return_value=torch.nn.Sequential(
                    torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
                ),
            ):
                inference_model = load_model(
                    Path(directory) / first.checkpoint_prefix / "latest.pt"
                )
            self.assertFalse(inference_model.training)
            sample = torch.ones(2, 2)
            with torch.inference_mode():
                torch.testing.assert_close(
                    inference_model(sample), resumed.model.eval()(sample)
                )

            # Exercise the CLI's built-in model, JSON input and file input.
            model = build_model().eval()
            checkpoint = Path(directory) / "inference.pt"
            torch.save(
                {
                    "format_version": 1,
                    "config": {"model_id": "synthetic-mlp"},
                    "model": model.state_dict(),
                },
                checkpoint,
            )
            features = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            with redirect_stdout(io.StringIO()) as output:
                infer_cli(
                    [
                        "infer",
                        "--checkpoint",
                        str(checkpoint),
                        "--input",
                        json.dumps(features),
                    ]
                )
            result = json.loads(output.getvalue())
            with torch.inference_mode():
                expected = model(torch.tensor([features])).softmax(-1)
            torch.testing.assert_close(torch.tensor(result["probabilities"]), expected)
            self.assertEqual(result["predictions"], expected.argmax(-1).tolist())
            input_file = Path(directory) / "inputs.json"
            input_file.write_text(json.dumps([features, features]))
            with redirect_stdout(io.StringIO()) as output:
                infer_cli(
                    [
                        "infer",
                        "--checkpoint",
                        str(checkpoint),
                        "--input-file",
                        str(input_file),
                    ]
                )
            self.assertEqual(len(json.loads(output.getvalue())["predictions"]), 2)
            with (
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as invalid,
            ):
                infer_cli(
                    ["infer", "--checkpoint", str(checkpoint), "--input", "[1, 2]"]
                )
            self.assertEqual(invalid.exception.code, 2)
            config_file = Path(directory) / "config.json"
            config_file.write_text(
                json.dumps({"device": "cpu:0", "precision": "bfloat16"})
            )
            with (
                patch(
                    "amarken.cli.run_inference", return_value={"predictions": [1]}
                ) as predict,
                redirect_stdout(io.StringIO()),
            ):
                infer_cli(
                    [
                        "infer",
                        "--config",
                        str(config_file),
                        "--checkpoint",
                        str(checkpoint),
                        "--input",
                        json.dumps(features),
                    ]
                )
            self.assertEqual(predict.call_args.kwargs["device"], "cpu:0")
            self.assertEqual(predict.call_args.kwargs["precision"], "bfloat16")

            # Final validation evaluates the checkpoint and weights uneven batches by samples.
            from amarken.modules.example import validation_step
            from amarken.validation import run_validation

            validation_inputs = torch.tensor([features, features, features])
            with torch.inference_mode():
                logits = model(validation_inputs)
            labels = logits.argmax(-1)
            labels[-1] = 1 - labels[-1]
            rows = TensorDataset(validation_inputs, labels)
            with (
                patch("amarken.validation.load_model", side_effect=load_model),
                patch(
                    "amarken.validation.validation_step", side_effect=validation_step
                ),
                patch("amarken.validation.build_dataset", return_value=(rows, rows)),
                patch(
                    "amarken.validation.build_dataloaders",
                    return_value=(None, DataLoader(rows, batch_size=2)),
                ),
            ):
                metrics = run_validation(checkpoint, TrainingConfig())
                self.assertEqual(metrics["samples"], 3)
                self.assertAlmostEqual(metrics["val/accuracy"], 2 / 3)
                self.assertAlmostEqual(
                    metrics["val/loss"],
                    torch.nn.functional.cross_entropy(logits, labels).item(),
                    places=6,
                )
            with (
                patch("amarken.cli.run_validation", return_value=metrics) as validate,
                redirect_stdout(io.StringIO()) as output,
            ):
                cli_main(
                    [
                        "validate",
                        "--config",
                        str(config_file),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )
            self.assertEqual(json.loads(output.getvalue()), metrics)
            self.assertEqual(validate.call_args.args[1].precision, "bfloat16")

            with (
                patch("amarken.validation.load_model", side_effect=load_model),
                patch(
                    "amarken.validation.validation_step", side_effect=validation_step
                ),
                patch("amarken.validation.build_dataset") as training_data,
                patch(
                    "amarken.validation.build_test_dataset", return_value=rows
                ) as test_data,
                patch(
                    "amarken.validation.build_dataloaders",
                    return_value=(None, DataLoader(rows, batch_size=2)),
                ),
            ):
                test_metrics = run_validation(
                    checkpoint, TrainingConfig(test_size=3), split="test"
                )
                training_data.assert_not_called()
                test_data.assert_called_once_with(seed=42, test_size=3)
                self.assertAlmostEqual(test_metrics["test/accuracy"], 2 / 3)
                self.assertNotIn("val/accuracy", test_metrics)
            with (
                patch(
                    "amarken.cli.run_validation", return_value=test_metrics
                ) as evaluate,
                redirect_stdout(io.StringIO()) as output,
            ):
                cli_main(
                    [
                        "test",
                        "--config",
                        str(config_file),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )
            self.assertEqual(json.loads(output.getvalue()), test_metrics)
            self.assertEqual(evaluate.call_args.kwargs["split"], "test")

            # Both implementations expose the same build/load/predict calls.
            self.assertEqual(predict_numeric(model, features), result)
            from transformers import AutoModel, Qwen3Config

            from amarken.modules import base

            architecture = Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=4096,
            )
            tiny = AutoModel.from_config(
                architecture, dtype=getattr(torch, base.BaseConfig().dtype)
            )
            tokenizer = Mock(
                pad_token_id=0,
                eos_token="<eos>",
                return_value={
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                },
            )
            with (
                patch(
                    "amarken.modules.base.AutoModel.from_pretrained", return_value=tiny
                ) as pretrained,
                patch(
                    "amarken.modules.base.AutoConfig.from_pretrained",
                    return_value=architecture,
                ),
                patch(
                    "amarken.modules.base.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
            ):
                with self.assertRaisesRegex(NotImplementedError, "task.*loss"):
                    base.loss_fn(None, None, "cpu")
                backbone = base.build_model(device="cpu")
                backbone.train()
                batch = backbone.tokenize(["hello"])
                optimizer = torch.optim.SGD(backbone.parameters(), lr=0.01)
                backbone(batch).sum().backward()
                with self.assertRaisesRegex(ValueError, "attention_mask"):
                    backbone({"input_ids": batch["input_ids"]})
                with self.assertRaisesRegex(TypeError, "tokenized dictionary"):
                    backbone(batch["input_ids"])
                optimizer.step()
                expected = base.predict(backbone, "hello")
                trainer = Trainer(
                    backbone, [], None, TrainingConfig(), optimizer=optimizer
                )
                with WandbLogger(
                    enabled=False,
                    checkpoints=CheckpointConfig(backend="local", async_save=True),
                ) as logger:
                    pending = logger.save_checkpoint(
                        trainer.state_dict(), "backbone.pt"
                    )
                    logger.flush_checkpoints()
                restored = base.load_model(pending.result(), device="cpu")
                self.assertEqual(pretrained.call_count, 1)
                self.assertFalse(restored.training)
                torch.testing.assert_close(
                    torch.tensor(base.predict(restored, ["hello"])["features"]),
                    torch.tensor(expected["features"]),
                )
                self.assertEqual(backbone.hidden_size, 16)
                with (
                    patch("amarken.inference.load_model", base.load_model),
                    patch("amarken.inference.predict", base.predict),
                ):
                    self.assertEqual(
                        run_inference(pending.result(), ["hello"]), expected
                    )

                # A task-specific derivative uses the same interface and its own loss.
                from amarken.modules import classification

                classifier = classification.build_model(device="cpu")
                self.assertTrue(classifier.head.training)
                self.assertFalse(classifier.backbone.training)
                self.assertTrue(
                    all(not p.requires_grad for p in classifier.backbone.parameters())
                )
                head_before = classifier.head.weight.detach().clone()
                backbone_before = {
                    name: p.detach().clone()
                    for name, p in classifier.backbone.named_parameters()
                }
                tokenized = classifier.tokenize(["hello"])
                classification_batch = (tokenized, torch.tensor([1]))
                training_config = TrainingConfig(
                    epochs=1,
                    run_name="classification-test",
                    checkpoints={"backend": "local", "async_save": True},
                )
                classification_trainer = Trainer(
                    classifier,
                    [classification_batch],
                    classification.loss_fn,
                    training_config,
                    validation_loader=[classification_batch],
                )
                history = classification_trainer.fit()
                self.assertTrue(torch.isfinite(torch.tensor(history[0]["train/loss"])))
                self.assertFalse(torch.equal(head_before, classifier.head.weight))
                for name, parameter in classifier.backbone.named_parameters():
                    self.assertTrue(torch.equal(backbone_before[name], parameter))
                expected_classes = classification.predict(classifier, "hello")
                classifier_checkpoint = (
                    Path(directory) / "classification-test/latest.pt"
                )
                restored_classifier = classification.load_model(classifier_checkpoint)
                actual_classes = classification.predict(restored_classifier, ["hello"])
                self.assertEqual(
                    actual_classes["predictions"], expected_classes["predictions"]
                )
                self.assertEqual(actual_classes["labels"], expected_classes["labels"])
                torch.testing.assert_close(
                    torch.tensor(actual_classes["probabilities"]),
                    torch.tensor(expected_classes["probabilities"]),
                )
                self.assertEqual(pretrained.call_count, 2)
                restored_bf16 = classification.load_model(
                    classifier_checkpoint, precision="bfloat16"
                )
                self.assertTrue(
                    all(p.dtype == torch.bfloat16 for p in restored_bf16.parameters())
                )
                numeric_bf16 = build_model(precision="bfloat16")
                self.assertTrue(
                    all(p.dtype == torch.bfloat16 for p in numeric_bf16.parameters())
                )

                with self.assertRaisesRegex(TypeError, "tokenized dictionary"):
                    classification.loss_fn(
                        classifier, (torch.ones(1, 8), torch.tensor([1])), "cpu"
                    )

                # Parallel dataset implementations: random tensors and HF text rows.
                from datasets import Dataset as HubDataset

                from amarken.datasets import example as random_data
                from amarken.datasets import huggingface as hub_data
                from amarken.main import train_example

                random_train, random_validation = random_data.build_dataset(
                    seed=42, train_size=12, val_size=4
                )
                self.assertEqual((len(random_train), len(random_validation)), (12, 4))
                repeated_train, _ = random_data.build_dataset(
                    seed=42, train_size=12, val_size=4
                )
                torch.testing.assert_close(
                    random_train.tensors[0], repeated_train.tensors[0]
                )
                random_loaders = random_data.build_dataloaders(
                    random_train, random_validation, batch_size=8
                )
                random_inputs, random_targets = next(iter(random_loaders[0]))
                self.assertEqual(tuple(random_inputs.shape), (8, 8))
                self.assertEqual(random_targets.dtype, torch.long)
                self.assertTrue(
                    torch.isfinite(
                        loss_fn(build_model(), (random_inputs, random_targets), "cpu")
                    )
                )

                text_rows = HubDataset.from_dict(
                    {
                        "sentence": ["good", "bad", "great", "awful"],
                        "label": [1, 0, 1, 0],
                    }
                )
                with (
                    patch(
                        "amarken.datasets.huggingface.load_dataset",
                        return_value=[text_rows, text_rows],
                    ) as hub_load,
                ):
                    text_train, text_validation = hub_data.build_dataset(
                        seed=42, train_size=3, val_size=2
                    )
                    self.assertEqual((len(text_train), len(text_validation)), (3, 2))
                    held_out = hub_data.build_test_dataset(seed=42, test_size=2)
                    self.assertTrue(
                        set(text_validation["sentence"]).isdisjoint(
                            held_out["sentence"]
                        )
                    )
                    changed_seed = hub_data.build_test_dataset(seed=99, test_size=2)
                    self.assertEqual(
                        set(held_out["sentence"]), set(changed_seed["sentence"])
                    )
                    self.assertEqual(
                        hub_load.call_args.kwargs["split"], ["train", "validation"]
                    )
                tokenizer.side_effect = lambda texts, **kwargs: {
                    "input_ids": torch.tensor([[1, 2, 3]] * len(texts)),
                    "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
                }
                text_loaders = hub_data.build_dataloaders(
                    text_train,
                    text_validation,
                    batch_size=2,
                    tokenize=restored_classifier.tokenize,
                )
                text_inputs, text_targets = next(iter(text_loaders[0]))
                self.assertEqual(tuple(text_inputs["input_ids"].shape), (2, 3))
                self.assertEqual(text_targets.dtype, torch.long)
                self.assertTrue(
                    torch.isfinite(
                        classification.loss_fn(
                            restored_classifier, (text_inputs, text_targets), "cpu"
                        )
                    )
                )
                with self.assertRaisesRegex(TypeError, "tokenize"):
                    hub_data.build_dataloaders(
                        text_train, text_validation, batch_size=2
                    )
                dataset_config = TrainingConfig(
                    epochs=1,
                    batch_size=2,
                    run_name="hf-dataset-test",
                    checkpoints={"backend": "local", "async_save": True},
                )
                with (
                    patch("amarken.main.build_model", return_value=restored_classifier),
                    patch(
                        "amarken.main.build_dataset",
                        return_value=(text_train, text_validation),
                    ),
                    patch("amarken.main.build_dataloaders", hub_data.build_dataloaders),
                    patch("amarken.main.loss_fn", classification.loss_fn),
                ):
                    prefix, dataset_history = train_example(dataset_config)
                self.assertEqual(prefix, "hf-dataset-test")
                self.assertTrue(
                    torch.isfinite(torch.tensor(dataset_history[0]["val/loss"]))
                )
                self.assertTrue((Path(directory) / prefix / "latest.pt").exists())

    def test_checkpoint_policies(self):
        nested = TrainingConfig(wandb={"enabled": True, "mode": "online"})
        self.assertTrue(nested.wandb.enabled)
        self.assertEqual(nested.wandb.mode, "online")
        self.assertFalse(TrainingConfig().wandb.enabled)
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

        # The entry point parses training options once and passes typed values.
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.json"
            config_file.write_text("{}")
            with (
                patch("amarken.cli.train_example", return_value=("test", [])) as train,
                redirect_stdout(io.StringIO()),
            ):
                cli_main(
                    ["example", "--config", str(config_file), "--resume", "saved.pt"]
                )
            self.assertIsInstance(train.call_args.args[0], TrainingConfig)
            self.assertEqual(train.call_args.kwargs["resume"], Path("saved.pt"))

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
