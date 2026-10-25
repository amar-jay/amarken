"""Candidate-conditioned scalar scoring with a shared Qwen encoder."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from amarken.base import BaseConfig, BaseModel
from amarken.constants import (
    ARTIFACT_VERSION,
    CANDIDATE_TEMPLATE,
    DATA_SCHEMA_VERSION,
    DECISION_KINDS,
    LORA_TARGETS,
    NOUL_TEMPLATE,
    PROMPT_VERSION,
)
from amarken.sampler import Sampler, SamplerConfig
from amarken.storage import fingerprint, read_json, write_json


@dataclass(frozen=True)
class DecisionConfig:
    candidate_chunk_size: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self):
        if self.candidate_chunk_size < 1 or self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("Chunk size and LoRA dimensions must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid LoRA dropout")


def validate_question(question):
    if question.get("kind") not in DECISION_KINDS:
        raise ValueError("Unknown decision kind")
    if (
        not isinstance(question.get("question"), str)
        or not question["question"].strip()
    ):
        raise ValueError("Question text must be nonempty")
    options = question.get("options", [])
    if not isinstance(options, (list, tuple)):
        raise TypeError("Options must be a list of labels")
    if question["kind"] == "noul":
        if options:
            raise ValueError("noul must not supply options")
    elif (
        len(options) < 2
        or any(not isinstance(o, str) or not o.strip() for o in options)
        or len(options) != len(set(options))
    ):
        raise ValueError("Options must contain at least two distinct nonempty labels")


def render_candidates(state, question):
    validate_question(question)
    if not isinstance(state, str) or not state.strip():
        raise ValueError("State must be nonempty text")
    # JSON escaping keeps newlines/delimiters inside state distinguishable from fields.
    fields = {
        "state": json.dumps(state, ensure_ascii=False),
        "question": json.dumps(question["question"], ensure_ascii=False),
    }
    if question["kind"] == "noul":
        return [NOUL_TEMPLATE.format(**fields)]
    return [
        CANDIDATE_TEMPLATE.format(**fields, candidate=json.dumps(o, ensure_ascii=False))
        for o in question["options"]
    ]


def decision_loss(logits, row):
    target = torch.as_tensor(row["target"], dtype=torch.float32, device=logits.device)
    if logits.shape != target.shape or not torch.isfinite(logits).all():
        raise ValueError("Logit/target shape mismatch or non-finite logits")
    if row["kind"] == "noul":
        return F.binary_cross_entropy_with_logits(logits.float(), target)
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum()


class DecisionModel(nn.Module):
    def __init__(self, base: BaseModel, config: DecisionConfig | None = None):
        super().__init__()
        self.base = base
        self.settings = config or DecisionConfig()
        self.head = nn.Linear(base.hidden_size, 1, bias=False).to(
            next(base.parameters()).device
        )
        self.calibration = {
            kind: {"temperature": 1.0, "status": "not_fitted"}
            for kind in DECISION_KINDS
        }

    def configure_training(self, mode="lora"):
        from peft import LoraConfig, get_peft_model

        if mode not in ("head", "lora"):
            raise ValueError("Training mode must be head or lora")
        self.base.requires_grad_(False)
        if mode == "lora":
            if not hasattr(self.base.model, "peft_config"):
                self.base.model = get_peft_model(
                    self.base.model,
                    LoraConfig(
                        r=self.settings.lora_rank,
                        lora_alpha=self.settings.lora_alpha,
                        lora_dropout=self.settings.lora_dropout,
                        target_modules=list(LORA_TARGETS),
                        task_type="FEATURE_EXTRACTION",
                        bias="none",
                    ),
                )
            else:
                for name, parameter in self.base.model.named_parameters():
                    parameter.requires_grad_("lora_" in name)
            self.base.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.head.requires_grad_(True)

    def score_texts(self, texts):
        features = self.base(**self.base.tokenize(texts))
        return self.head(features.to(self.head.weight.dtype)).squeeze(-1)

    def candidate_chunks(self, state, question):
        texts = render_candidates(state, question)
        # Keep padding, batch position and low-precision kernels independent of
        # caller option order. Restore caller order before loss or sampling.
        order = sorted(range(len(texts)), key=texts.__getitem__)
        inverse = sorted(range(len(order)), key=order.__getitem__)
        texts = [texts[i] for i in order]
        size = self.settings.candidate_chunk_size
        return [texts[i : i + size] for i in range(0, len(texts), size)], order, inverse

    def forward(self, state, question):
        chunks, _, inverse = self.candidate_chunks(state, question)
        return torch.cat([self.score_texts(chunk) for chunk in chunks])[inverse]

    def backward_question(self, row, scale=1.0):
        """Exact two-pass gradients with RNG replay, including LoRA dropout."""
        chunks, order, inverse = self.candidate_chunks(row["state"], row)
        states, values = [], []
        device = next(self.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.no_grad():
            for chunk in chunks:
                states.append(
                    (
                        torch.get_rng_state(),
                        torch.cuda.get_rng_state(device) if devices else None,
                    )
                )
                values.append(self.score_texts(chunk))
        logits = torch.cat(values)[inverse].detach().requires_grad_(True)
        loss = decision_loss(logits, row)
        (gradient,) = torch.autograd.grad(loss * scale, logits)
        gradient = gradient[order]
        offset = 0
        # Restore the state after the first pass when exiting, not after replay.
        with torch.random.fork_rng(devices=devices):
            for chunk, (cpu_rng, cuda_rng) in zip(chunks, states):
                torch.set_rng_state(cpu_rng)
                if devices:
                    torch.cuda.set_rng_state(cuda_rng, device)
                output = self.score_texts(chunk)
                output.backward(gradient[offset : offset + len(chunk)])
                offset += len(chunk)
        return float(loss.detach())

    @torch.inference_mode()
    def predict(self, state, questions, sampler_config=None):
        if not isinstance(questions, list) or not questions:
            raise ValueError("questions must be a nonempty list")
        ids = [q.get("id") for q in questions]
        if any(not isinstance(i, str) or not i.strip() for i in ids) or len(
            set(ids)
        ) != len(ids):
            raise ValueError("Each question needs a distinct nonempty string ID")
        config = sampler_config or SamplerConfig()
        results = {}
        sampler = Sampler(config)
        was_training = self.training
        self.eval()
        try:
            for q in questions:
                validate_question(q)
                temperature = self.calibration[q["kind"]]["temperature"]
                # Calibration and exploratory sampling temperatures are explicit multipliers.
                sampler.config = SamplerConfig(
                    config.mode, config.temperature * temperature, config.seed
                )
                results[q["id"]] = sampler.sample(
                    self(state, q).float().cpu().tolist(),
                    kind=q["kind"],
                    options=q.get("options", []),
                    score_values=q.get("score_values"),
                )
                results[q["id"]]["calibration_status"] = self.calibration[q["kind"]][
                    "status"
                ]
        finally:
            self.train(was_training)
        return results

    def save(self, directory, *, embed_base=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        adapter = hasattr(self.base.model, "peft_config")
        if adapter:
            self.base.model.save_pretrained(directory / "adapter")
        if embed_base:
            if adapter:
                from copy import deepcopy

                deepcopy(self.base.model).unload().save_pretrained(directory / "base")
            else:
                self.base.model.save_pretrained(directory / "base")
        self.base.tokenizer.save_pretrained(directory / "tokenizer")
        torch.save(self.head.state_dict(), directory / "head.pt")
        write_json(
            directory / "model.json",
            {
                "artifact_version": ARTIFACT_VERSION,
                "schema_version": DATA_SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "base": asdict(self.base.settings),
                "decision": asdict(self.settings),
                "adapter": adapter,
                "embedded_base": embed_base,
                "training_provenance": getattr(self, "training_provenance", None),
            },
        )
        write_json(directory / "calibration.json", self.calibration)
        write_json(
            directory / "checksums.json",
            {
                str(p.relative_to(directory)): fingerprint(p)
                for p in directory.rglob("*")
                if p.is_file()
            },
        )

    @classmethod
    def load(cls, directory, *, device="cpu", dtype="float32", trainable=False):
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        directory = Path(directory)
        for name, digest in read_json(directory / "checksums.json").items():
            if fingerprint(directory / name) != digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        meta = read_json(directory / "model.json")
        if (
            meta["artifact_version"],
            meta["schema_version"],
            meta["prompt_version"],
        ) != (ARTIFACT_VERSION, DATA_SCHEMA_VERSION, PROMPT_VERSION):
            raise ValueError("Incompatible artifact, data schema or prompt version")
        settings = BaseConfig(**{**meta["base"], "device": device, "dtype": dtype})
        tokenizer = AutoTokenizer.from_pretrained(
            directory / "tokenizer", local_files_only=True
        )
        source = directory / "base" if meta["embedded_base"] else settings.model_id
        kwargs = (
            {}
            if meta["embedded_base"]
            else {"revision": settings.revision, "cache_dir": settings.cache_dir}
        )
        model = AutoModel.from_pretrained(
            source, dtype=getattr(torch, dtype), trust_remote_code=False, **kwargs
        ).to(device)
        if meta["adapter"]:
            model = PeftModel.from_pretrained(
                model, directory / "adapter", is_trainable=trainable
            )
        instance = cls(
            BaseModel(model, tokenizer, settings), DecisionConfig(**meta["decision"])
        )
        instance.head.load_state_dict(
            torch.load(directory / "head.pt", map_location=device, weights_only=True)
        )
        instance.calibration = read_json(directory / "calibration.json")
        instance.training_provenance = meta.get("training_provenance")
        return instance.eval()
