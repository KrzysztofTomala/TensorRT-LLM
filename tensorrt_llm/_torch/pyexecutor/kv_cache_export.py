from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from .llm_request import LlmRequest
from .resource_manager import ResourceManager, ResourceManagerType


@dataclass
class KvCacheLayerSnapshot:
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    metadata: dict[str, Any]


@dataclass
class RequestKvCacheSnapshot:
    request_id: int
    token_ids: list[int]
    layers: list[KvCacheLayerSnapshot]
    metadata: dict[str, Any]

    @property
    def tensors(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        return {
            layer.layer_idx: (layer.key, layer.value)
            for layer in self.layers
        }


def _safe_scalar_attr(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
    except Exception as exc:
        return f"<error: {type(exc).__name__}: {exc}>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return int(value)
    except Exception:
        return str(value)


def _get_kv_cache_manager(resource_manager: ResourceManager) -> Any:
    kv_cache_manager = resource_manager.get_resource_manager(
        ResourceManagerType.KV_CACHE_MANAGER)
    if kv_cache_manager is None:
        raise RuntimeError("KV cache manager is not available")
    return kv_cache_manager


def _layer_indices(kv_cache_manager: Any) -> list[int]:
    layer_offsets = getattr(kv_cache_manager, "layer_offsets", None)
    if isinstance(layer_offsets, dict) and layer_offsets:
        return sorted(int(layer_idx) for layer_idx in layer_offsets)
    num_layers = int(getattr(kv_cache_manager, "num_layers", 0) or 0)
    return list(range(num_layers))


def _block_ids_for_layer(kv_cache_manager: Any, request: LlmRequest,
                         layer_idx: int) -> list[int]:
    request_id = int(request.py_request_id)
    try:
        block_ids = kv_cache_manager.get_batch_cache_indices(
            [request_id], layer_idx=layer_idx)[0]
    except TypeError:
        block_ids = kv_cache_manager.get_batch_cache_indices([request_id])[0]
    except Exception:
        block_ids = kv_cache_manager.get_cache_indices(request)
    return [int(block_id) for block_id in block_ids if int(block_id) >= 0]


def gather_layer_kv_cache(
    kv_cache_manager: Any,
    request: LlmRequest,
    *,
    layer_idx: int,
    seq_len: int,
    kv_layout: str = "NHD",
    clone: bool = False,
) -> KvCacheLayerSnapshot:
    block_ids = _block_ids_for_layer(kv_cache_manager, request, layer_idx)
    if not block_ids:
        raise RuntimeError(f"No live KV blocks for request {request.py_request_id}")

    buffer = kv_cache_manager.get_buffers(layer_idx, kv_layout=kv_layout)
    if buffer is None:
        raise RuntimeError(f"KV cache manager returned no buffer for layer {layer_idx}")

    page_idx = torch.tensor(block_ids, dtype=torch.long, device=buffer.device)
    pages = buffer.index_select(0, page_idx)
    if kv_layout == "NHD":
        flat = pages.permute(0, 2, 1, 3, 4).reshape(
            -1, pages.shape[1], pages.shape[3], pages.shape[4])
    elif kv_layout == "HND":
        flat = pages.permute(0, 3, 1, 2, 4).reshape(
            -1, pages.shape[1], pages.shape[2], pages.shape[4])
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")

    flat = flat[:seq_len]
    if flat.shape[1] < 2:
        raise RuntimeError(
            f"Expected K/V factor >= 2, got shape {tuple(flat.shape)}")

    key = flat[:, 0].permute(1, 0, 2).unsqueeze(0).contiguous()
    value = flat[:, 1].permute(1, 0, 2).unsqueeze(0).contiguous()
    if clone:
        key = key.clone()
        value = value.clone()

    metadata = {
        "block_ids": block_ids,
        "buffer_shape": list(buffer.shape),
        "gathered_tokens": int(flat.shape[0]),
        "tokens_per_block": int(getattr(kv_cache_manager, "tokens_per_block")),
        "kv_layout": kv_layout,
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
    }
    return KvCacheLayerSnapshot(
        layer_idx=layer_idx,
        key=key,
        value=value,
        metadata=metadata,
    )


def export_request_kv_cache(
    resource_manager: ResourceManager,
    request: LlmRequest,
    *,
    kv_layout: str = "NHD",
    layers: Optional[list[int]] = None,
    max_layers: Optional[int] = None,
    clone: bool = False,
) -> RequestKvCacheSnapshot:
    if max_layers is not None and max_layers < 1:
        raise ValueError("max_layers must be >= 1 when set")

    kv_cache_manager = _get_kv_cache_manager(resource_manager)
    token_ids = list(request.get_tokens(0))
    layer_indices = layers if layers is not None else _layer_indices(kv_cache_manager)
    if max_layers is not None:
        layer_indices = layer_indices[:max_layers]

    layer_snapshots = [
        gather_layer_kv_cache(
            kv_cache_manager,
            request,
            layer_idx=int(layer_idx),
            seq_len=len(token_ids),
            kv_layout=kv_layout,
            clone=clone,
        ) for layer_idx in layer_indices
    ]
    metadata = {
        "request": {
            name: _safe_scalar_attr(request, name)
            for name in (
                "py_request_id",
                "py_prompt_len",
                "py_orig_prompt_len",
                "py_max_new_tokens",
                "py_decoding_iter",
                "max_beam_num_tokens",
                "context_current_position",
                "state",
                "is_finished",
            )
        },
        "token_count": len(token_ids),
        "token_tail": token_ids[-16:],
        "kv_cache_manager": {
            "class": type(kv_cache_manager).__name__,
            "tokens_per_block": _safe_scalar_attr(kv_cache_manager,
                                                  "tokens_per_block"),
            "kv_factor": _safe_scalar_attr(kv_cache_manager, "kv_factor"),
            "head_dim": _safe_scalar_attr(kv_cache_manager, "head_dim"),
            "num_layers": len(_layer_indices(kv_cache_manager)),
            "num_kv_heads_per_layer": [
                int(x)
                for x in list(
                    getattr(kv_cache_manager, "num_kv_heads_per_layer", []))[:8]
            ],
        },
        "layers": [layer.metadata | {"layer_idx": layer.layer_idx}
                   for layer in layer_snapshots],
    }
    return RequestKvCacheSnapshot(
        request_id=int(request.py_request_id),
        token_ids=token_ids,
        layers=layer_snapshots,
        metadata=metadata,
    )
