"""Optional Ming speech example: frozen codec/LLM, flow conditioning + stop training.

Ming predicts continuous audio latents, not discrete class/token labels. Each
utterance contributes equally to the loss, regardless of recording length.
"""

import io
import subprocess
import sys
from contextlib import nullcontext
from math import gcd
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from amarken.constants import (
    HUB_CACHE,
    MING_PROMPT,
    MING_RUNTIME,
    MING_RUNTIME_REVISION,
    REVISIONS,
)
from amarken.utils.distributed import primary
from amarken.utils.helpers import load_checkpoint_weights, prepare_training_precision


def _runtime():
    if not (MING_RUNTIME / "modeling_bailingmm.py").is_file():
        raise RuntimeError(
            "Install the optional Ming runtime; see README.md's TTS example"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(MING_RUNTIME), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MING_RUNTIME_REVISION:
        raise RuntimeError(f"Ming runtime must be pinned to {MING_RUNTIME_REVISION}")
    if str(MING_RUNTIME) not in sys.path:
        sys.path.insert(0, str(MING_RUNTIME))
    from configuration_bailingmm import BailingMMConfig
    from modeling_bailingmm import BailingMMNativeForConditionalGeneration

    return BailingMMConfig, BailingMMNativeForConditionalGeneration


def _spectral_float(module, inputs, output):
    # Upstream complex construction/FFT does not support BF16 on CPU.
    return output.float()


class MingTTS(nn.Module):
    def __init__(self, core, *, precision="bfloat16"):
        super().__init__()
        self.core = core.requires_grad_(False)
        core.audio.decoder.head.out.register_forward_hook(_spectral_float)
        # A deliberately small real adaptation objective, suitable for a demo.
        core.flowloss.cfm.model.c_embedder.cond_embedder.requires_grad_(True)
        core.stop_head.requires_grad_(True)
        self.precision = precision
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.core.eval()
        self.core.flowloss.train(mode)
        self.core.stop_head.train(mode)
        return self

    def _autocast(self):
        device = next(self.parameters()).device
        # Frozen weights retain the checkpoint's native dtype during loading.
        dtype = self.core.get_input_embeddings().weight.dtype
        if dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device.type, dtype=dtype)

    def tokenize(self, samples):
        """Decode/resample whole paired recordings; no audio/text truncation."""
        import soundfile as sf
        from scipy.signal import resample_poly

        result = []
        for sample in samples:
            audio = sample["audio"]
            source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
            waveform, rate = sf.read(source, dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
            target_rate = self.core.config.audio_tokenizer_config.sample_rate
            factor = gcd(rate, target_rate)
            waveform = resample_poly(waveform, target_rate // factor, rate // factor)
            text = sample["text"]
            if not isinstance(text, str) or not text.strip() or waveform.size == 0:
                raise ValueError("TTS samples require nonempty text and audio")
            tensor = torch.from_numpy(waveform.copy()).float()
            if not torch.isfinite(tensor).all():
                raise ValueError("Audio must contain finite samples")
            result.append((text, tensor))
        return {"utterances": result, "batch_size": len(result)}

    def forward(self, batch):
        losses, flows, stops, accuracies = [], [], [], []
        core = self.core
        device = next(core.parameters()).device
        with self._autocast():
            for text, waveform in batch["utterances"]:
                with torch.no_grad():
                    wave = waveform.to(device).unsqueeze(0)
                    latents, lengths = core.audio.encode_latent(
                        wave, torch.tensor([wave.shape[1]], device=device)
                    )
                    # The codec uses inference_mode internally; clone for autograd.
                    target = latents[:, : int(lengths[0])].clone()
                    _, prefix = core.prepare_input_embed(MING_PROMPT, text)
                    previous = (
                        core.linear_proj_audio(
                            target[:, :-1].transpose(0, 1)
                        ).transpose(0, 1)
                        if target.shape[1] > 1
                        else prefix[:, :0]
                    )
                    embeddings = torch.cat([prefix, previous], dim=1)
                    hidden = core.model.model(
                        inputs_embeds=embeddings, use_cache=False
                    ).last_hidden_state[:, prefix.shape[1] - 1 :]
                target = target.transpose(0, 1)  # [frames, 1, latent_dim]
                history = torch.cat([torch.zeros_like(target[:1]), target[:-1]], dim=0)
                cond = hidden.transpose(0, 1).clone()
                mask = torch.ones(target.shape[:2], device=device)
                if self.training:
                    flow = core.flowloss(cond, target, history, mask, core.patch_size)
                else:
                    # Fixed midpoint/zero-noise diagnostic rather than random val loss.
                    prediction = core.flowloss.cfm.model(
                        x=target * 0.5,
                        t=torch.full((len(target),), 0.5, device=device),
                        c=cond,
                        latent_history=history,
                        mask=mask.bool(),
                    )[:, -core.patch_size :]
                    flow = F.mse_loss(prediction.float(), target.float())
                labels = torch.zeros(len(target), dtype=torch.long, device=device)
                labels[-1] = 1
                logits = core.stop_head(cond).float().squeeze(1)
                stop = F.cross_entropy(logits, labels)
                losses.append(flow + stop)
                flows.append(flow.detach())
                stops.append(stop.detach())
                accuracies.append((logits.argmax(-1) == labels).float().mean())
        return {
            "loss": torch.stack(losses).mean(),
            "flow_mse": torch.stack(flows).mean(),
            "stop_loss": torch.stack(stops).mean(),
            "stop_accuracy": torch.stack(accuracies).mean(),
        }


def build_model(*, model_id=None, device="cpu", precision="float32"):
    from transformers import Qwen2Tokenizer

    if not model_id or model_id not in REVISIONS:
        raise ValueError(
            "Set model_id in the config and add its pinned revision to REVISIONS"
        )
    revision = REVISIONS[model_id]
    config_class, model_class = _runtime()
    config = config_class.from_pretrained(
        model_id, revision=revision, cache_dir=HUB_CACHE
    )
    core = model_class.from_pretrained(
        model_id,
        revision=revision,
        config=config,
        cache_dir=HUB_CACHE,
        dtype="auto",
    )
    if primary():
        print("Loading Ming tokenizer...", file=sys.stderr, flush=True)
    core.tokenizer = Qwen2Tokenizer.from_pretrained(
        model_id, revision=revision, cache_dir=HUB_CACHE
    )
    return MingTTS(core, precision=precision).to(device)


def load_model(
    checkpoint: str | Path, *, model_id=None, device="cpu", precision="float32"
):
    model = build_model(model_id=model_id, precision=precision)
    model.load_state_dict(load_checkpoint_weights(checkpoint), strict=True, assign=True)
    prepare_training_precision(model, precision)
    return model.to(device).eval()


def loss_fn(model, batch, device):
    return model(batch)["loss"]


def build_metrics():
    from torchmetrics import MeanMetric, MetricCollection

    return MetricCollection(
        {key: MeanMetric() for key in ("flow_mse", "stop_loss", "stop_accuracy")},
        compute_groups=False,
    )


def validation_step(model, batch, device, metrics):
    result = model(batch)
    for key in ("flow_mse", "stop_loss", "stop_accuracy"):
        metrics[key].update(result[key], weight=batch["batch_size"])
    return {"loss": result["loss"].item()}


def fsdp_modules(model):
    # Preprocessing invokes frozen submodules outside forward. Do not silently
    # promise safe collective ordering until a multi-GPU TTS run is verified.
    raise ValueError(
        "Ming TTS example currently supports single-device training and DDP"
    )


def predict(model, inputs):
    """Generate a WAV from {text, output_file, max_steps?}; no reference voice needed."""
    import soundfile as sf

    if not isinstance(inputs, dict) or not isinstance(inputs.get("text"), str):
        raise TypeError("TTS input requires text and output_file")
    text = inputs["text"].strip()
    output = inputs.get("output_file")
    steps = inputs.get("max_steps", 250)
    if not text or not isinstance(output, str) or not output:
        raise ValueError("TTS input requires nonempty text and output_file")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 5 <= steps <= 1000:
        raise ValueError("max_steps must be an integer between 5 and 1000")
    model.eval()
    core = model.core
    stopped, cache, chunks = False, None, []
    with torch.inference_mode(), model._autocast():
        _, embedding = core.prepare_input_embed(MING_PROMPT, text)
        history = embedding.new_zeros(1, core.history_patch_size, core.latent_dim)
        for step in range(steps):
            result = core.model.model(
                inputs_embeds=embedding, past_key_values=cache, use_cache=True
            )
            cache = result.past_key_values
            condition = result.last_hidden_state[:, -1:]
            latent = core.flowloss.sample(condition, history, cfg=2.0)
            chunks.append(latent)
            if (
                core.stop_head(condition).float().softmax(-1)[0, 0, 1] > 0.5
                and step > 3
            ):
                stopped = True
                break
            embedding = core.linear_proj_audio(latent)
            history = torch.cat([history, latent], dim=1)[:, -core.history_patch_size :]
        audio = core.audio.decode(torch.cat(chunks, dim=1), last_chunk=True)[0]
    audio = audio.detach().float().cpu().reshape(-1).numpy()
    if not torch.isfinite(torch.from_numpy(audio)).all() or not len(audio):
        raise ValueError("Generated waveform must be nonempty and finite")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = core.config.audio_tokenizer_config.sample_rate
    sf.write(path, audio, sample_rate)
    return {
        "output_file": str(path),
        "sample_rate": sample_rate,
        "samples": len(audio),
        "stopped": stopped,
    }
