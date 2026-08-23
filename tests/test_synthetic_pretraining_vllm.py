import json
import re
from types import SimpleNamespace
from src.distillation.synthetic_pretraining import make_spec
from src.distillation.synthetic_pretraining_vllm import (
    engine_kwargs,
    generation_conversation,
    run,
    sampling_kwargs,
)


def test_a100_engine_is_one_text_only_continuously_batched_model():
    config = json.load(open("configs/synthetic_pretraining_1m_a100.json"))
    kwargs = engine_kwargs(config)
    assert kwargs["model"] == "Qwen/Qwen3.5-2B" and kwargs["tensor_parallel_size"] == 1
    assert (
        kwargs["language_model_only"] is True
        and kwargs["max_num_seqs"] == 256
        and kwargs["max_num_batched_tokens"] == 16384
    )


def test_a100_80gb_profile_matches_host_and_expands_scheduler_capacity():
    config = json.load(open("configs/synthetic_pretraining_1m_a100_80gb.json"))
    kwargs = engine_kwargs(config)
    assert config["host_profile"] == {
        "gpu": "NVIDIA A100 80GB",
        "gpu_vram_gb": 80,
        "system_ram_gb": 120,
        "vcpus": 28,
    }
    assert config["request_batch_size"] == 8192
    assert kwargs["tensor_parallel_size"] == 1 and kwargs["max_num_seqs"] == 512
    assert (
        kwargs["max_num_batched_tokens"] == 32768
        and kwargs["gpu_memory_utilization"] == 0.92
    )
    assert kwargs["dtype"] == "bfloat16"


def test_private_grounding_is_not_saved_in_training_system_message():
    spec = next(
        make_spec(i, 91423) for i in range(100) if make_spec(i, 91423)["required"]
    )
    generation = generation_conversation(spec)
    assert all(value in generation[0]["content"] for value in spec["required"])
    assert all(
        value not in spec["messages"][0]["content"] for value in spec["required"]
    )


def test_retries_are_deterministic():
    config = json.load(open("configs/synthetic_pretraining_1m_a100.json"))
    assert sampling_kwargs(config, 7, False)["temperature"] == 1.0
    assert sampling_kwargs(config, 7, True)["temperature"] == 0.0
    assert sampling_kwargs(config, 7, True)["seed"] == 7


def test_offline_batch_run_writes_resumable_records_without_vllm_installed(tmp_path):
    config = json.load(open("configs/synthetic_pretraining_1m_a100.json"))
    config.update(
        {
            "target_accepted": 12,
            "request_batch_size": 5,
            "shard_size": 4,
            "output_dir": str(tmp_path / "out"),
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        def chat(self, conversations, **_kwargs):
            outputs = []
            for conversation in conversations:
                system = conversation[0]["content"]
                match = re.search(r"names: (.*?)\. Do not", system)
                values = match.group(1) if match else ""
                if "Turkish" in system:
                    text = f"Yanıt {values or 'kısa ve yararlı'} olur."
                else:
                    text = f"The answer is {values or 'short and useful'}."
                candidate = SimpleNamespace(
                    text=text,
                    finish_reason="stop",
                    stop_reason=None,
                    token_ids=[1, 2, 3],
                )
                outputs.append(
                    SimpleNamespace(outputs=[candidate], prompt_token_ids=[1, 2])
                )
            return outputs

    run(
        config_path,
        output_override=tmp_path / "out",
        llm=FakeLLM(),
        sampling_params_type=Params,
    )
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert manifest["accepted"] == 12 and manifest["complete"] is True
    assert len(list((tmp_path / "out/shards").glob("shard-*.jsonl"))) == 3
