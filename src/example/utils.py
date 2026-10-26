_LAYER_IDX_REQUIRED_MSG = (
    "{cls} requires `layer_idx` when `past_key_values` is provided."
)


def require_cache_layer_idx(module, past_key_values):
    layer_idx = getattr(module, "layer_idx", None)
    if past_key_values is not None and layer_idx is None:
        raise ValueError(_LAYER_IDX_REQUIRED_MSG.format(cls=module.__class__.__name__))
    return layer_idx


def get_layer_cache(module, past_key_values):
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None and len(past_key_values) > layer_idx:
        return past_key_values[layer_idx]
    return None


def update_layer_cache(module, past_key_values, **kwargs):
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None:
        return past_key_values.update(layer_idx=layer_idx, **kwargs)
    return None
