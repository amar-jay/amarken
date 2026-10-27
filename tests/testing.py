import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from amarken.cli import main as cli_main
from amarken.datasets.config import DatasetConfig
from amarken.inference import run_inference
from amarken.logger import CheckpointConfig, WandbLogger
from amarken.modules.example import build_model, load_model, loss_fn
from amarken.modules.example import predict as predict_numeric
from amarken.trainer import Trainer, TrainingConfig, seed_everything
from amarken.utils import prompt_run_name


def make_config(*, dataset=None, **options):
    return TrainingConfig(
        model_id="synthetic-mlp",
        dataset={"id": "stanfordnlp/sst2"} if dataset is None else dataset,
        **options,
    )


class TrainerTests(unittest.TestCase):
    def make_trainer(self, epochs):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4), torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)
        )
        inputs = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 24
        targets = torch.arange(12) % 2
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=True)
        config = make_config(
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
        from amarken.distributed import launched, resolve_strategy, training_process

        for rank_value in (None, "-1"):
            with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
                if rank_value is None:
                    os.environ.pop("RANK", None)
                else:
                    os.environ["RANK"] = rank_value
                self.assertFalse(launched())
                for strategy in ("auto", "ddp", "fsdp2"):
                    self.assertEqual(resolve_strategy(strategy), "single")
                with patch("torch.distributed.init_process_group") as initialize:
                    with training_process("cpu", strategy="ddp"):
                        self.assertEqual(self.make_trainer(1).strategy, "single")
                    initialize.assert_not_called()
                with WandbLogger(enabled=False) as logger:
                    self.assertEqual(logger.rank, 0)
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"}):
            self.assertTrue(launched())
            self.assertEqual(resolve_strategy("auto"), "ddp")

        for local_rank in (0, 1):
            with (
                patch.dict(
                    os.environ,
                    {
                        "WORLD_SIZE": "2",
                        "LOCAL_WORLD_SIZE": "2",
                        "LOCAL_RANK": str(local_rank),
                        "RANK": str(local_rank),
                    },
                ),
                patch("torch.cuda.device_count", return_value=1),
                patch("torch.cuda.set_device") as set_device,
                patch("torch.distributed.init_process_group") as initialize,
                self.assertRaisesRegex(ValueError, "only 1 CUDA devices are visible"),
                training_process("cuda"),
            ):
                self.fail("Oversized GPU launch should fail before entering training")
            set_device.assert_not_called()
            initialize.assert_not_called()

        for bad_precision in ("float16",):
            with (
                patch.dict(os.environ, {"RANK": "0"}),
                self.assertRaisesRegex(ValueError, "float16"),
            ):
                from amarken.distributed import resolve_strategy

                resolve_strategy("fsdp2", bad_precision)
        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch("torch.__version__", "2.6.0"),
            self.assertRaisesRegex(ValueError, "2.14"),
        ):
            resolve_strategy("fsdp2")
        with tempfile.TemporaryDirectory() as fsdp_directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            environment["OMP_NUM_THREADS"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    str(Path(__file__).resolve()),
                    "--fsdp-worker",
                    fsdp_directory,
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            load_model(Path(fsdp_directory) / "checkpoints" / "full" / "latest.pt")

        with tempfile.TemporaryDirectory() as distributed_directory:
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            environment["OMP_NUM_THREADS"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=2",
                    str(Path(__file__).resolve()),
                    "--ddp-worker",
                    distributed_directory,
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            reports = [
                json.loads(
                    (Path(distributed_directory) / f"rank-{rank}.json").read_text()
                )
                for rank in range(2)
            ]
            self.assertEqual(reports[0]["history"], reports[1]["history"])
            self.assertEqual(reports[0]["prompts"], 2)
            self.assertEqual(reports[1]["prompts"], 0)
            self.assertEqual(reports[0]["loggers"], 3)
            self.assertEqual(reports[1]["loggers"], 0)
            self.assertTrue(
                set(reports[0]["indices"]).isdisjoint(reports[1]["indices"])
            )
            self.assertEqual(
                len(reports[0]["indices"]) + len(reports[1]["indices"]), 32
            )
            # DDP weights remain loadable by the existing numeric inference module.
            weights = (
                Path(distributed_directory)
                / "checkpoints"
                / "full"
                / "shared"
                / "latest.pt"
            )
            load_model(weights)
            with self.assertRaisesRegex(ValueError, "same world size"):
                self.make_trainer(3).resume(weights)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
            patch("amarken.inference.load_model", load_model),
            patch("amarken.inference.predict", predict_numeric),
        ):
            inference_config = Path(directory) / "inference-config.json"
            inference_config.write_text(
                json.dumps(
                    {"model_id": "synthetic-mlp", "dataset": {"id": "stanfordnlp/sst2"}}
                )
            )

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
                json.dumps(
                    {
                        "model_id": "synthetic-mlp",
                        "dataset": {"id": "stanfordnlp/sst2"},
                        "device": "cpu:0",
                        "precision": "bfloat16",
                    }
                )
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
                metrics = run_validation(checkpoint, make_config())
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
                        "--format",
                        "json",
                        "--config",
                        str(config_file),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )
            self.assertEqual(json.loads(output.getvalue()), metrics)
            self.assertEqual(validate.call_args.args[1].precision, "bfloat16")
            from amarken.utils import format_validation_table

            table = format_validation_table(metrics)
            self.assertIn("Metric", table)
            self.assertIn("val/accuracy", table)
            self.assertIn("Row / Col", table)
            self.assertEqual(table, format_validation_table(json.dumps(metrics)))
            with (
                patch("amarken.cli.run_validation", return_value=metrics),
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
            self.assertEqual(output.getvalue().rstrip(), table)

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
                    checkpoint,
                    make_config(dataset={"id": "stanfordnlp/sst2", "test_size": 3}),
                    split="test",
                )
                training_data.assert_not_called()
                test_data.assert_called_once_with(
                    DatasetConfig(id="stanfordnlp/sst2", test_size=3), seed=42
                )
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
                        "--format",
                        "json",
                        "--config",
                        str(config_file),
                        "--checkpoint",
                        str(checkpoint),
                    ]
                )
            self.assertEqual(json.loads(output.getvalue()), test_metrics)
            self.assertEqual(evaluate.call_args.kwargs["split"], "test")

            # Compute non-linear metrics over the split, independent of batch size.
            from amarken.modules.example import build_metrics

            metric_inputs = torch.tensor(
                [[4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]]
            )
            metric_targets = torch.tensor([0, 0, 1, 1, 1])
            metric_rows = TensorDataset(metric_inputs, metric_targets)
            results = []
            for batch_size in (2, 5):
                with (
                    patch(
                        "amarken.validation.load_model",
                        return_value=torch.nn.Identity(),
                    ),
                    patch(
                        "amarken.validation.build_metrics", side_effect=build_metrics
                    ),
                    patch(
                        "amarken.validation.build_dataset",
                        return_value=(metric_rows, metric_rows),
                    ),
                    patch(
                        "amarken.validation.build_dataloaders",
                        return_value=(
                            None,
                            DataLoader(metric_rows, batch_size=batch_size),
                        ),
                    ),
                ):
                    results.append(run_validation(checkpoint, make_config()))
            for output_metrics in results:
                self.assertEqual(
                    output_metrics["val/confusion_matrix"], [[1, 1], [1, 2]]
                )
                self.assertAlmostEqual(output_metrics["val/accuracy"], 0.6)
                self.assertAlmostEqual(output_metrics["val/f1_macro"], 7 / 12)
                self.assertAlmostEqual(output_metrics["val/f1_weighted"], 0.6)
                self.assertAlmostEqual(output_metrics["val/precision_macro"], 7 / 12)
                self.assertAlmostEqual(output_metrics["val/recall_macro"], 7 / 12)
                self.assertAlmostEqual(output_metrics["val/f1_per_class"][0], 0.5)
                self.assertAlmostEqual(output_metrics["val/f1_per_class"][1], 2 / 3)
                self.assertAlmostEqual(
                    output_metrics["val/calibration_error"],
                    metric_inputs.softmax(-1)[0].max().item() - 0.6,
                    places=6,
                )
                json.dumps(output_metrics, allow_nan=False)
            for key in results[0]:
                if key == "val/loss":
                    self.assertAlmostEqual(results[0][key], results[1][key], places=6)
                else:
                    self.assertEqual(results[0][key], results[1][key])
            single = build_metrics()
            single.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
            self.assertTrue(
                all(torch.isfinite(value).all() for value in single.compute().values())
            )

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
                    backbone, [], None, make_config(), optimizer=optimizer
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
                training_config = make_config(
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
                from amarken.main import run_train

                random_train, random_validation = random_data.build_dataset(
                    DatasetConfig(id="synthetic", train_size=12, val_size=4), seed=42
                )
                self.assertEqual((len(random_train), len(random_validation)), (12, 4))
                repeated_train, _ = random_data.build_dataset(
                    DatasetConfig(id="synthetic", train_size=12, val_size=4), seed=42
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
                        DatasetConfig(
                            id=make_config().dataset.id, train_size=3, val_size=2
                        ),
                        seed=42,
                    )
                    self.assertEqual((len(text_train), len(text_validation)), (3, 2))
                    self.assertEqual(
                        hub_load.call_args.args[0], make_config().dataset.id
                    )
                    with patch.dict(
                        hub_data.DATASET_REVISIONS,
                        {"custom/reviews": "pinned-revision"},
                    ):
                        hub_data.build_dataset(
                            DatasetConfig(
                                id="custom/reviews",
                                config="reviews-subset",
                                train_size=3,
                                val_size=2,
                            )
                        )
                        self.assertEqual(hub_load.call_args.args[0], "custom/reviews")
                        self.assertEqual(
                            hub_load.call_args.kwargs["name"], "reviews-subset"
                        )
                        self.assertEqual(
                            hub_load.call_args.kwargs["revision"], "pinned-revision"
                        )
                        hub_data.build_test_dataset(
                            DatasetConfig(
                                id="custom/reviews",
                                config="reviews-subset",
                                test_size=2,
                            )
                        )
                        self.assertEqual(hub_load.call_args.args[0], "custom/reviews")
                        self.assertEqual(
                            hub_load.call_args.kwargs["name"], "reviews-subset"
                        )
                        self.assertEqual(
                            hub_load.call_args.kwargs["revision"], "pinned-revision"
                        )

                    held_out = hub_data.build_test_dataset(
                        DatasetConfig(id=make_config().dataset.id, test_size=2), seed=42
                    )
                    self.assertTrue(
                        set(text_validation["sentence"]).isdisjoint(
                            held_out["sentence"]
                        )
                    )
                    changed_seed = hub_data.build_test_dataset(
                        DatasetConfig(id=make_config().dataset.id, test_size=2), seed=99
                    )
                    self.assertEqual(
                        set(held_out["sentence"]), set(changed_seed["sentence"])
                    )
                    self.assertEqual(
                        hub_load.call_args.kwargs["split"], ["train", "validation"]
                    )
                    ratio_train, ratio_val = hub_data.build_dataset(
                        DatasetConfig(id="stanfordnlp/sst2", test_fraction=0.25)
                    )
                    ratio_test = hub_data.build_test_dataset(
                        DatasetConfig(id="stanfordnlp/sst2", test_fraction=0.25)
                    )
                    self.assertEqual(
                        (len(ratio_train), len(ratio_val), len(ratio_test)), (4, 3, 1)
                    )
                    self.assertTrue(
                        set(ratio_val["sentence"]).isdisjoint(ratio_test["sentence"])
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
                dataset_config = make_config(
                    epochs=1,
                    batch_size=2,
                    run_name="hf-dataset-test",
                    dataset={
                        "id": "stanfordnlp/sst2",
                        "config": "reviews-subset",
                        "train_size": 3,
                        "val_size": 2,
                        "test_size": 1,
                        "test_fraction": 0.25,
                    },
                    precision="bfloat16",
                    checkpoints={"backend": "local", "async_save": True},
                )
                with (
                    patch(
                        "amarken.main.build_model", return_value=restored_classifier
                    ) as model_builder,
                    patch(
                        "amarken.main.build_dataset",
                        return_value=(text_train, text_validation),
                    ) as dataset_builder,
                    patch("amarken.main.build_dataloaders", hub_data.build_dataloaders),
                    patch("amarken.main.loss_fn", classification.loss_fn),
                ):
                    prefix, dataset_history = run_train(dataset_config)
                dataset_builder.assert_called_once_with(dataset_config.dataset, seed=42)
                model_builder.assert_called_once_with(device="cpu", precision="float32")
                self.assertTrue(
                    all(
                        p.dtype == torch.float32
                        for p in restored_classifier.head.parameters()
                    )
                )
                self.assertTrue(
                    all(
                        p.dtype == torch.bfloat16
                        for p in restored_classifier.backbone.parameters()
                    )
                )
                # Mixed storage also supports direct prediction without autocast.
                self.assertEqual(
                    len(
                        classification.predict(restored_classifier, ["good"])[
                            "predictions"
                        ]
                    ),
                    1,
                )
                mixed_checkpoint = torch.load(
                    Path(directory) / prefix / "latest.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(
                    mixed_checkpoint["model"]["head.weight"].dtype, torch.float32
                )
                self.assertTrue(
                    all(
                        state["exp_avg"].dtype == torch.float32
                        for state in mixed_checkpoint["optimizer"]["state"].values()
                    )
                )
                self.assertEqual(prefix, "hf-dataset-test")
                self.assertTrue(
                    torch.isfinite(torch.tensor(dataset_history[0]["val/loss"]))
                )
                self.assertTrue((Path(directory) / prefix / "latest.pt").exists())

    def test_checkpoint_policies(self):
        self.assertIsNone(make_config().dataset.config)
        self.assertEqual(
            make_config(
                dataset={"id": "stanfordnlp/sst2", "config": "subset"}
            ).dataset.config,
            "subset",
        )
        with self.assertRaisesRegex(ValueError, "dataset.config"):
            make_config(dataset={"id": "stanfordnlp/sst2", "config": 1})
        for fraction in (0, 1, -0.1, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "test_fraction"):
                make_config(
                    dataset={"id": "stanfordnlp/sst2", "test_fraction": fraction}
                )
        from dataclasses import asdict

        round_trip = TrainingConfig(
            **asdict(
                make_config(dataset={"id": "stanfordnlp/sst2", "test_fraction": 0.25})
            )
        )
        self.assertEqual(round_trip.dataset.test_fraction, 0.25)
        nested = make_config(wandb={"enabled": True, "mode": "online"})
        self.assertTrue(nested.wandb.enabled)
        self.assertEqual(nested.wandb.mode, "online")
        self.assertFalse(make_config().wandb.enabled)
        self.assertEqual(make_config().checkpoints.policy, "default")
        self.assertTrue(CheckpointConfig().overwrite)
        with self.assertRaises(ValueError):
            CheckpointConfig(policy="all")
        with self.assertRaises(ValueError):
            CheckpointConfig(overwrite="false")
        config = make_config(checkpoints={"backend": "local", "overwrite": False})
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("questionary.text") as prompt,
        ):
            prompt.return_value.ask.return_value = "my-run"
            self.assertEqual(prompt_run_name(config), "my-run")
            validator = prompt.call_args.kwargs["validate"]
            self.assertIs(validator("my-run"), True)
            self.assertIsInstance(validator("../escape"), str)
            with (
                tempfile.TemporaryDirectory() as existing_root,
                patch("amarken.logger.CHECKPOINT_DIR", Path(existing_root)),
            ):
                existing = Path(existing_root) / config.run_name / "my-run"
                existing.mkdir(parents=True)
                self.assertIn("already exists", validator("my-run"))
                self.assertIs(validator("another-run"), True)

            prompt.return_value.ask.return_value = "  "
            self.assertIsNone(prompt_run_name(config))
            prompt.return_value.ask.return_value = None
            with self.assertRaises(SystemExit) as cancelled:
                prompt_run_name(config)
            self.assertEqual(cancelled.exception.code, 130)
            prompt.reset_mock()
            self.assertIsNone(prompt_run_name(config, resume=True))
            self.assertIsNone(prompt_run_name(make_config()))
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

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("amarken.logger.CHECKPOINT_DIR", Path(directory)),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "amarken.utils.prompt_run_name", side_effect=["taken", "available"]
            ) as prompt,
            redirect_stderr(io.StringIO()) as feedback,
        ):
            existing = Path(directory) / "test/taken"
            existing.mkdir(parents=True)
            marker = existing / "keep.txt"
            marker.write_text("previous run")
            trainer = self.make_trainer(1)
            trainer.config.checkpoints = CheckpointConfig(
                backend="local", overwrite=False
            )
            with patch.object(trainer, "_fit", return_value=[]):
                trainer.fit()
            self.assertEqual(prompt.call_count, 2)
            self.assertEqual(trainer.run_label, "available")
            self.assertTrue((Path(directory) / "test/available").is_dir())
            self.assertEqual(marker.read_text(), "previous run")
            self.assertIn("already exists", feedback.getvalue())

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
            config_file.write_text(
                json.dumps(
                    {"model_id": "synthetic-mlp", "dataset": {"id": "stanfordnlp/sst2"}}
                )
            )
            with (
                patch("amarken.cli.run_train", return_value=("test", [])) as train,
                redirect_stdout(io.StringIO()),
            ):
                cli_main(
                    ["train", "--config", str(config_file), "--resume", "saved.pt"]
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


def distributed_worker(directory):
    """Two-process integration exercise, launched only by the existing test."""
    from amarken import distributed
    from amarken.datasets.example import build_dataloaders, build_dataset
    from amarken.utils import load_checkpoint_weights

    training_losses = []

    def objective(model, batch, device):
        inputs, targets = (item.to(device) for item in batch)
        logits = model(inputs)
        if model.training:
            logits = logits + torch.randn_like(logits) * 0.1
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if model.training:
            training_losses.append((loss.detach().item(), len(targets)))
        return loss

    directory = Path(directory)
    with distributed.training_process("cpu"):  # noqa: SIM117 - verify cleanup below
        with (
            patch("amarken.logger.CHECKPOINT_DIR", directory / "checkpoints"),
            patch("amarken.utils.prompt_run_name", return_value="shared") as prompt,
            patch("amarken.trainer.WandbLogger", wraps=WandbLogger) as logger_factory,
        ):

            def create(name, epochs):
                seed_everything(42)
                model = build_model()
                train_data, validation_data = build_dataset(
                    DatasetConfig(id="synthetic", train_size=32, val_size=7)
                )
                train, validation = build_dataloaders(
                    train_data, validation_data, batch_size=3
                )
                config = make_config(
                    epochs=epochs,
                    run_name=name,
                    checkpoints=CheckpointConfig(
                        backend="local", overwrite=False, async_save=True
                    ),
                )
                return Trainer(
                    model, train, objective, config, validation_loader=validation
                )

            full = create("full", 2)
            history = full.fit()
            indices = list(full.train_loader.sampler)
            local_losses = training_losses[-len(full.train_loader) :]
            gathered_losses = [None] * distributed.world_size()
            torch.distributed.all_gather_object(gathered_losses, local_losses)
            loss_sum = sum(
                value * count for losses in gathered_losses for value, count in losses
            )
            sample_count = sum(
                count for losses in gathered_losses for _, count in losses
            )
            assert abs(history[-1]["train/loss"] - loss_sum / sample_count) < 1e-7

            # Validation covers all seven rows, including the short final batch.
            validation = full.validation_loader.dataset
            with torch.inference_mode():
                expected = objective(
                    distributed.unwrap(full.model).eval(), validation.tensors, "cpu"
                ).item()
            assert abs(history[-1]["val/loss"] - expected) < 1e-6
            parameters = torch.cat(
                [p.detach().flatten() for p in full.model.parameters()]
            )
            reference = parameters.clone()
            torch.distributed.broadcast(reference, 0)
            torch.testing.assert_close(parameters, reference, rtol=0, atol=0)
            partial = create("partial", 1)
            partial.fit()
            path = directory / "checkpoints" / "partial" / "shared" / "latest.pt"
            resumed = create("partial", 2)
            resumed.resume(path)
            resumed_history = resumed.fit()
            assert resumed_history[-1] == history[-1]
            for key, value in distributed.unwrap(full.model).state_dict().items():
                torch.testing.assert_close(
                    value,
                    distributed.unwrap(resumed.model).state_dict()[key],
                    rtol=0,
                    atol=0,
                )
            checkpoint = directory / "checkpoints" / "full" / "shared" / "latest.pt"
            assert all(
                not key.startswith("module.")
                for key in load_checkpoint_weights(checkpoint)
            )
            # A rank-zero failure must reach every worker without hanging.
            try:
                distributed.primary_call(
                    lambda: (_ for _ in ()).throw(ValueError("expected failure"))
                )
            except RuntimeError as error:
                assert "expected failure" in str(error)
            else:
                raise AssertionError("Rank-zero failure was not propagated")
            (directory / f"rank-{distributed.rank()}.json").write_text(
                json.dumps(
                    {
                        "history": history,
                        "indices": indices,
                        "prompts": prompt.call_count,
                        "loggers": logger_factory.call_count,
                    }
                )
            )
    assert not distributed.active()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--ddp-worker":
    distributed_worker(sys.argv[2])


def fsdp_worker(directory):
    """Real FSDP2 collectives, checkpoint export/resume and frozen HF groups."""
    from torch.distributed.tensor import DTensor

    from amarken import distributed
    from amarken.datasets.example import build_dataloaders, build_dataset
    from amarken.modules.example import fsdp_modules

    directory = Path(directory)
    device = os.environ.get("AMARKEN_TEST_DEVICE", "cpu")

    def objective(model, batch, device):
        inputs, targets = (value.to(device) for value in batch)
        logits = model(inputs).float()
        if model.training:
            logits = logits + torch.randn_like(logits) * 0.1
        return torch.nn.functional.cross_entropy(logits, targets)

    with (
        distributed.training_process(device, strategy="fsdp2") as local_device,
        patch("amarken.logger.CHECKPOINT_DIR", directory / "checkpoints"),
    ):

        def create(name, epochs, strategy="fsdp2", precision="float32"):
            seed_everything(42)
            model = build_model(precision=precision)
            train_rows, val_rows = build_dataset(
                DatasetConfig(id="synthetic", train_size=32, val_size=7)
            )
            train, val = build_dataloaders(train_rows, val_rows, batch_size=3)
            config = make_config(
                strategy=strategy,
                device=local_device,
                precision=precision,
                epochs=epochs,
                run_name=name,
                checkpoints=CheckpointConfig(backend="local", async_save=True),
            )
            return Trainer(
                model,
                train,
                objective,
                config,
                validation_loader=val,
                fsdp_modules=fsdp_modules,
            )

        full = create("full", 2)
        assert all(isinstance(p, DTensor) for p in full.model.parameters())
        if distributed.world_size() > 1:
            assert sum(p.to_local().numel() for p in full.model.parameters()) < sum(
                p.numel() for p in full.model.parameters()
            )
        history = full.fit()
        partial = create("partial", 1)
        partial_history = partial.fit()
        resumed = create("partial", 2)
        resumed.resume(directory / "checkpoints" / "partial" / "latest.pt")
        resumed_history = resumed.fit()
        assert resumed_history[-1] == history[-1]
        full_state, full_optimizer = distributed.full_state(full.model, full.optimizer)
        resumed_state, resumed_optimizer = distributed.full_state(
            resumed.model, resumed.optimizer
        )
        if not distributed.primary():
            assert not full_state and not full_optimizer
        if distributed.primary():
            for name in full_state:
                torch.testing.assert_close(
                    full_state[name], resumed_state[name], rtol=0, atol=0
                )
            for name in full_optimizer["state"]:
                for key in full_optimizer["state"][name]:
                    torch.testing.assert_close(
                        full_optimizer["state"][name][key],
                        resumed_optimizer["state"][name][key],
                        rtol=0,
                        atol=0,
                    )
            restored = load_model(
                directory / "checkpoints" / "full" / "latest.pt",
                device=local_device,
            )
            rows = full.validation_loader.dataset
            with torch.inference_mode():
                expected = objective(restored, rows.tensors, local_device).item()
            assert abs(expected - history[-1]["val/loss"]) < 1e-6
        reference = create("reference", 1, strategy="ddp")
        reference_history = reference.fit()
        assert (
            abs(reference_history[0]["train/loss"] - partial_history[0]["train/loss"])
            < 1e-5
        )
        partial_state, _ = distributed.full_state(partial.model, partial.optimizer)
        for name, param in distributed.unwrap(reference.model).state_dict().items():
            if distributed.primary():
                torch.testing.assert_close(
                    partial_state[name], param.cpu(), rtol=1e-5, atol=1e-6
                )
        try:
            full.resume(directory / "checkpoints" / "reference" / "latest.pt")
        except (RuntimeError, ValueError) as error:
            assert "same strategy" in str(error)
        else:
            raise AssertionError("Strategy mismatch should fail")
        try:
            full.resume(directory / "missing.pt")
        except RuntimeError as error:
            assert "FileNotFoundError" in str(error)
        else:
            raise AssertionError("Missing checkpoint should fail on all ranks")
        failing = create("write-failure", 1)
        with patch.object(
            WandbLogger,
            "save_checkpoint",
            side_effect=OSError("expected disk failure"),
        ):
            try:
                failing.fit()
            except RuntimeError as error:
                assert "expected disk failure" in str(error)
            else:
                raise AssertionError("Checkpoint write failure must reach every rank")
        try:
            Trainer(
                build_model(),
                [],
                objective,
                make_config(strategy="fsdp2", device=local_device),
                optimizer=Mock(),
                fsdp_modules=fsdp_modules,
            )
        except ValueError as error:
            assert "external optimizers" in str(error)
        else:
            raise AssertionError("External optimizers must be rejected before sharding")
        # Exercise reduced-precision collectives without downloading a model.
        bf16 = create("bf16", 1, precision="bfloat16")
        assert all(p.dtype == torch.float32 for p in bf16.model.parameters())
        bf16.fit()
        _, moments = distributed.full_state(bf16.model, bf16.optimizer)
        if distributed.primary():
            assert all(
                state["exp_avg"].dtype == torch.float32
                for state in moments["state"].values()
            )
        if device == "cuda":
            half = create("fp16", 1, strategy="ddp", precision="float16")
            assert half.scaler.is_enabled()
            assert all(p.dtype == torch.float32 for p in half.model.parameters())
            before_half = next(half.model.parameters()).detach().clone()
            half.fit()
            assert not torch.equal(before_half, next(half.model.parameters()))
            assert all(
                state["exp_avg"].dtype == torch.float32
                for state in half.optimizer.state.values()
            )
            restored_half = create("fp16", 2, strategy="ddp", precision="float16")
            restored_half.resume(directory / "checkpoints" / "fp16" / "latest.pt")
            assert restored_half.scaler.state_dict() == half.scaler.state_dict()
            restored_half.fit()

        from transformers import AutoModel, Qwen3Config

        from amarken.modules.base import BaseConfig, BaseModel
        from amarken.modules.classification import TextClassifier
        from amarken.modules.classification import fsdp_modules as classifier_groups
        from amarken.modules.classification import loss_fn as classifier_loss

        seed_everything(42)
        architecture = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        backbone = BaseModel(AutoModel.from_config(architecture), Mock(), BaseConfig())
        classifier = TextClassifier(backbone)
        from amarken.utils import prepare_training_precision

        prepare_training_precision(classifier, "bfloat16")
        before = {
            key: value.detach().clone()
            for key, value in classifier.state_dict().items()
        }
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
        batch = (inputs, torch.tensor([0, 1]))
        config = make_config(
            strategy="fsdp2",
            device=local_device,
            precision="bfloat16",
            epochs=1,
            run_name="classifier",
        )
        trainer = Trainer(
            classifier,
            [batch],
            classifier_loss,
            config,
            validation_loader=[batch],
            fsdp_modules=classifier_groups,
        )
        assert all(p.dtype == torch.float32 for p in trainer.model.head.parameters())
        assert all(
            p.dtype == torch.bfloat16 for p in trainer.model.backbone.parameters()
        )
        trainer.fit()
        weights, _ = distributed.full_state(trainer.model, trainer.optimizer)
        if distributed.primary():
            assert not torch.equal(weights["head.weight"], before["head.weight"])
            for key, value in before.items():
                if key.startswith("backbone."):
                    torch.testing.assert_close(weights[key], value, rtol=0, atol=0)
    assert not distributed.active()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--fsdp-worker":
    fsdp_worker(sys.argv[2])
