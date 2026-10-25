"""Amarken decision model tooling."""


def main() -> None:
    import argparse
    import json
    import torch
    from amarken.base import BaseConfig, BaseModel
    from amarken.constants import BASES

    parser = argparse.ArgumentParser(description="Load and smoke-test the Qwen model")
    parser.add_argument("--fallback", action="store_true", help="Explicitly select Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = BaseConfig(model_id=BASES[1] if args.fallback else BASES[0],
                            device=args.device, dtype=args.dtype)
    model = BaseModel.load(config, local_files_only=args.local_files_only)
    with torch.inference_mode():
        features = model(**model.tokenize(["State: payment failed. Question: Is action needed?"]))
    if not torch.isfinite(features).all():
        raise RuntimeError("Model produced non-finite features")
    print(json.dumps({"model": config.model_id, "revision": config.revision,
                      "feature_shape": list(features.shape), "dtype": str(features.dtype),
                      "device": str(features.device), "parameters": sum(p.numel() for p in model.parameters())}))
