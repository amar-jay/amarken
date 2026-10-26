"""Small torchrun helpers; task modules remain ordinary PyTorch code."""

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if active() else 0


def world_size() -> int:
    return dist.get_world_size() if active() else 1


def primary() -> bool:
    return rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def primary_call(callback):
    """All ranks participate; propagate rank-zero failures before continuing."""
    if not active():
        return callback()
    result = [None]
    if primary():
        try:
            result[0] = (True, callback())
        except BaseException as error:  # noqa: BLE001 - share cancellation
            result[0] = (False, f"{type(error).__name__}: {error}")
    dist.broadcast_object_list(result, src=0)
    success, value = result[0]
    if not success:
        raise RuntimeError(f"Rank zero failed: {value}")
    return value


def train_sampler(dataset):
    return DistributedSampler(dataset, shuffle=True) if active() else None


@contextmanager
def training_process(device: str):
    """Initialize before building models, and destroy only groups we own."""
    owned = False
    resolved = torch.device(device)
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            if resolved.type not in {"cpu", "cuda"}:
                raise ValueError("DDP supports cpu or cuda devices")
            if resolved.type == "cuda":
                local_rank = int(os.environ["LOCAL_RANK"])
                local_workers = int(
                    os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
                )
                gpu_count = torch.cuda.device_count()
                if local_workers > gpu_count or not 0 <= local_rank < gpu_count:
                    raise ValueError(
                        f"DDP requested {local_workers} local GPU workers, but only "
                        f"{gpu_count} CUDA devices are visible. Use --nproc-per-node "
                        "no greater than the GPU count, or run amarken train directly "
                        "for single-GPU training."
                    )
                if resolved.index is not None and resolved.index != local_rank:
                    raise ValueError(
                        "Use device=cuda under torchrun, without a fixed GPU index"
                    )
                torch.cuda.set_device(local_rank)
                resolved = torch.device("cuda", local_rank)
            if not active():
                dist.init_process_group(
                    backend="nccl" if resolved.type == "cuda" else "gloo",
                    timeout=timedelta(minutes=30),
                )
                owned = True
        yield str(resolved)
    finally:
        if owned:
            dist.destroy_process_group()
