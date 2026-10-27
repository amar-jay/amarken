"""Small torchrun helpers; task modules remain ordinary PyTorch code."""

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


def launched() -> bool:
    """torchrun sets RANK; WORLD_SIZE alone does not identify a launch."""
    return int(os.environ.get("RANK", "-1")) != -1


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return (
        dist.get_rank() if active() else (int(os.environ["RANK"]) if launched() else 0)
    )


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


def resolve_strategy(requested: str, precision: str = "float32") -> str:
    if requested not in {"auto", "ddp", "fsdp2"}:
        raise ValueError("strategy must be auto, ddp or fsdp2")
    if not launched() and not active():
        return "single"
    if requested == "auto":
        return "ddp"
    if requested == "fsdp2":
        if precision not in {"float32", "bfloat16"}:
            raise ValueError(
                "FSDP2 supports float32 and bfloat16; float16 is not supported"
            )
        if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 14):
            raise ValueError("FSDP2 requires PyTorch 2.14 or newer")
        try:
            from torch.distributed.fsdp import fully_shard  # noqa: F401
        except ImportError as error:
            raise ValueError(
                "Install PyTorch 2.14 or newer with FSDP2 support"
            ) from error
    return requested


def shard_model(model, groups, device, precision):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    modules = list(groups(model))
    if (
        not modules
        or modules[-1] is not model
        or len({id(m) for m in modules}) != len(modules)
    ):
        raise ValueError(
            "fsdp_modules must return unique groups bottom-up, ending with the model"
        )
    mesh = init_device_mesh(device.type, (world_size(),))
    policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, precision), reduce_dtype=torch.float32
    )
    for module in modules:
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    return model


def full_state(model, optimizer):
    """Collective snapshot; only rank zero receives complete CPU tensors."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    return get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )


def restore_full_state(model, optimizer, model_state, optimizer_state):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    # Separate APIs ensure nonzero ranks with empty dictionaries still load both.
    set_model_state_dict(model, model_state, options=options)
    set_optimizer_state_dict(model, optimizer, optimizer_state, options=options)


@contextmanager
def training_process(
    device: str, *, strategy: str = "auto", precision: str = "float32"
):
    """Initialize before building models, and destroy only groups we own."""
    selected = resolve_strategy(strategy, precision)
    owned = False
    resolved = torch.device(device)
    try:
        if selected != "single":
            if resolved.type not in {"cpu", "cuda"}:
                raise ValueError("Distributed training supports cpu or cuda devices")
            if resolved.type == "cuda":
                local_rank = int(os.environ["LOCAL_RANK"])
                local_workers = int(
                    os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
                )
                gpu_count = torch.cuda.device_count()
                if local_workers > gpu_count or not 0 <= local_rank < gpu_count:
                    raise ValueError(
                        f"Distributed training requested {local_workers} local GPU workers, but only "
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
